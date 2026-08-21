import json
import logging
import os
import sys
import time
from concurrent import futures
from datetime import datetime, timedelta, timezone

import grpc
import nltk
import proto.vocab_pb2
import proto.vocab_pb2_grpc
from fsrs import Card, Rating, Scheduler, State
from nltk.stem import WordNetLemmatizer


def ensure_nltk_data():
    required_resources = (
        ("tokenizers/punkt", "punkt"),
        ("corpora/wordnet", "wordnet"),
        ("taggers/averaged_perceptron_tagger_eng", "averaged_perceptron_tagger_eng"),
    )

    for lookup_path, package_name in required_resources:
        try:
            nltk.data.find(lookup_path)
        except LookupError:
            nltk.download(package_name, quiet=True)


class VocabServiceServicer(proto.vocab_pb2_grpc.VocabServiceServicer):
    _DEFAULT_TAIL_WEIGHTS = (0.5425, 0.0912, 0.0658, 0.1542)

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        load_config()
        ensure_nltk_data()
        self._lemmatizer = WordNetLemmatizer()
        self.logger.info("VocabServiceServicer initialized")

    def _scheduler_from_request(self, request):
        kwargs = {}
        if request.HasField("request_retention") and 0 < request.request_retention.value <= 1:
            kwargs["desired_retention"] = request.request_retention.value
        if request.HasField("maximum_interval") and request.maximum_interval.value > 0:
            kwargs["maximum_interval"] = request.maximum_interval.value
        if request.w and len(request.w) >= 17:
            params = list(request.w)
            if len(params) == 17:
                params.extend(self._DEFAULT_TAIL_WEIGHTS)
            if len(params) >= 21:
                kwargs["parameters"] = tuple(params[:21])
        if request.learning_steps_seconds:
            kwargs["learning_steps"] = tuple(
                timedelta(seconds=int(s)) for s in request.learning_steps_seconds
            )
        if request.relearning_steps_seconds:
            kwargs["relearning_steps"] = tuple(
                timedelta(seconds=int(s)) for s in request.relearning_steps_seconds
            )
        if request.HasField("enable_fuzzing"):
            kwargs["enable_fuzzing"] = bool(request.enable_fuzzing)
        return Scheduler(**kwargs) if kwargs else Scheduler()

    def AnalyzeText(self, request, context):
        try:
            response = proto.vocab_pb2.AnalyzeTextResponse()
            for token in self._analyze_text(request.text):
                item = response.tokens.add()
                item.text = token["text"]
                item.type = token["type"]
                if token.get("lemma"):
                    item.lemma.value = token["lemma"]
                if token.get("pos_tag"):
                    item.pos_tag.value = token["pos_tag"]
            return response
        except Exception as e:
            self.logger.error("Error analyzing text: %s", str(e), exc_info=True)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Error analyzing text: {str(e)}")
            return proto.vocab_pb2.AnalyzeTextResponse()

    def AnalyzeTargetWord(self, request, context):
        try:
            target_lower = (request.target_word or "").strip().lower()
            for token in self._analyze_text(request.sentence):
                if token["type"] == proto.vocab_pb2.WORD and token["text"].lower() == target_lower:
                    response = proto.vocab_pb2.AnalyzeTargetWordResponse(
                        lemma=token.get("lemma", target_lower)
                    )
                    if token.get("pos_tag"):
                        response.pos_tag.value = token["pos_tag"]
                    return response

            return proto.vocab_pb2.AnalyzeTargetWordResponse(
                lemma=self._lemmatize_word(request.target_word, "NN")
            )
        except Exception as e:
            self.logger.error("Error analyzing target word: %s", str(e), exc_info=True)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Error analyzing target word: {str(e)}")
            return proto.vocab_pb2.AnalyzeTargetWordResponse()

    def ReviewCard(self, request, context):
        start_time = time.time()
        try:
            step_val = request.card.step.value if request.card.HasField("step") else 0
            stability_val = (
                request.card.stability.value if request.card.HasField("stability") else 0.0
            )
            difficulty_val = (
                request.card.difficulty.value if request.card.HasField("difficulty") else 0.0
            )

            self.logger.info(
                f"ReviewCard request received: quality={request.quality}, "
                f"card_state={request.card.state}, step={step_val}, "
                f"stability={stability_val}, difficulty={difficulty_val}"
            )

            scheduler = self._scheduler_from_request(request)
            mapped_card = Card()

            raw_state = int(request.card.state)
            if raw_state == 0:
                mapped_card.state = State.Learning
            else:
                try:
                    mapped_card.state = State(raw_state)
                except ValueError:
                    mapped_card.state = State.Learning

            # Передаём step с wire как есть (в т.ч. 0 для первого learning-шага); иначе py-fsrs не продвигает лестницу шагов.
            if request.card.HasField("step"):
                mapped_card.step = request.card.step.value
            else:
                mapped_card.step = 0

            if request.card.HasField("stability") and request.card.stability.value > 0:
                mapped_card.stability = request.card.stability.value

            if request.card.HasField("difficulty") and request.card.difficulty.value > 0:
                mapped_card.difficulty = request.card.difficulty.value

            if request.HasField("review_at") and request.review_at is not None:
                review_datetime = request.review_at.ToDatetime().replace(tzinfo=timezone.utc)
            else:
                review_datetime = datetime.now(timezone.utc)

            if request.card.HasField("due"):
                mapped_card.due = request.card.due.ToDatetime().replace(tzinfo=timezone.utc)
            else:
                mapped_card.due = review_datetime

            if request.card.HasField("last_review"):
                mapped_card.last_review = request.card.last_review.ToDatetime().replace(
                    tzinfo=timezone.utc
                )
            else:
                mapped_card.last_review = review_datetime

            rating = Rating(request.quality)
            review_duration = (
                request.review_duration.value
                if request.HasField("review_duration") and request.review_duration.value > 0
                else 0
            )

            self.logger.debug(
                f"Processing FSRS review: state={mapped_card.state}, "
                f"due={mapped_card.due}, last_review={mapped_card.last_review}, "
                f"rating={rating}, review_duration={review_duration}, "
                f"stability={mapped_card.stability}, difficulty={mapped_card.difficulty}"
            )

            card, review_log = scheduler.review_card(
                mapped_card, rating, review_datetime, review_duration
            )

            response = proto.vocab_pb2.ReviewResponse()
            response.card.state = int(card.state)
            response.card.step.value = card.step if card.step is not None else 0
            response.card.stability.value = card.stability if card.stability is not None else 0.0
            response.card.difficulty.value = card.difficulty if card.difficulty is not None else 0.0
            response.card.due = card.due
            response.card.last_review = card.last_review

            response.review_log.rating = int(review_log.rating)
            response.review_log.review_datetime = review_log.review_datetime
            if isinstance(review_log.review_duration, timedelta):
                response.review_log.review_duration.value = int(
                    review_log.review_duration.total_seconds() * 1000
                )
            else:
                response.review_log.review_duration.value = (
                    int(review_log.review_duration)
                    if review_log.review_duration is not None
                    else 0
                )

            elapsed_time = (time.time() - start_time) * 1000
            stability_str = f"{card.stability:.2f}" if card.stability is not None else "None"
            difficulty_str = f"{card.difficulty:.2f}" if card.difficulty is not None else "None"
            step_str = str(card.step) if card.step is not None else "None"
            self.logger.info(
                f"ReviewCard processed successfully: new_state={card.state}, "
                f"new_step={step_str}, new_stability={stability_str}, "
                f"new_difficulty={difficulty_str}, elapsed={elapsed_time:.2f}ms"
            )

            return response
        except Exception as e:
            elapsed_time = (time.time() - start_time) * 1000
            self.logger.error(
                f"Error processing ReviewCard: {str(e)}, elapsed={elapsed_time:.2f}ms",
                exc_info=True,
            )
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Error review card: {str(e)}")
            return proto.vocab_pb2.ReviewResponse()


def setup_logging(log_level=logging.INFO):
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"

    logging.basicConfig(
        level=log_level,
        format=log_format,
        datefmt=date_format,
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logging.getLogger("grpc").setLevel(logging.WARNING)


def load_config(path="config.json"):
    logger = logging.getLogger(__name__)
    if not os.path.exists(path):
        logger.error(f"Config file not found at: {path}")
        raise FileNotFoundError(f"Config file not found at: {path}")

    logger.info(f"Loading config from: {path}")
    with open(path, "r") as f:
        config = json.load(f)
    logger.info(f"Config loaded successfully: {config}")
    return config


def serve():
    setup_logging()
    logger = logging.getLogger(__name__)

    try:
        config = load_config()
        port = config.get("server_port", 40051)
        address = f"[::]:{port}"

        logger.info(f"Initializing gRPC server on {address}")
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
        proto.vocab_pb2_grpc.add_VocabServiceServicer_to_server(
            VocabServiceServicer(), server
        )
        server.add_insecure_port(address)

        logger.info(f"FSRS gRPC service starting on port {port}...")
        server.start()
        logger.info("Server started successfully and ready to accept requests")

        try:
            while True:
                time.sleep(86400)
        except KeyboardInterrupt:
            logger.info("Received shutdown signal (KeyboardInterrupt)")
            server.stop(0)
            logger.info("Server stopped gracefully")
    except Exception as e:
        logger.critical(f"Fatal error starting server: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    serve()
