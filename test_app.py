import json
from io import BytesIO
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
from pypdf import PdfWriter

import app as document_app


class AppTests(unittest.TestCase):
    def setUp(self):
        document_app.app.config["TESTING"] = True
        self.client = document_app.app.test_client()
        self.saved_documents = dict(document_app.documents)
        document_app.documents.clear()
        self.data_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.data_dir.cleanup)
        self.index_dir_patch = patch.object(
            document_app, "INDEX_DIR", Path(self.data_dir.name)
        )
        self.index_dir_patch.start()
        self.addCleanup(self.index_dir_patch.stop)
        self.addCleanup(self.restore_saved_documents)

    def restore_saved_documents(self):
        document_app.documents.clear()
        document_app.documents.update(self.saved_documents)

    def test_upload_rejects_non_pdf(self):
        response = self.client.post(
            "/api/upload",
            data={"pdf": (BytesIO(b"not a pdf"), "notes.txt")},
            content_type="multipart/form-data",
        )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json["error"], "Please choose a PDF file.")

    def test_load_chunks_rejects_encrypted_pdf(self):
        writer = PdfWriter()
        writer.add_blank_page(width=100, height=100)
        writer.encrypt("secret")
        stream = BytesIO()
        writer.write(stream)
        stream.seek(0)

        with self.assertRaisesRegex(document_app.PdfUploadError, "encrypted"):
            document_app.load_chunks(stream)

    def test_load_chunks_returns_no_text_for_scanned_or_blank_pdf(self):
        writer = PdfWriter()
        writer.add_blank_page(width=100, height=100)
        stream = BytesIO()
        writer.write(stream)
        stream.seek(0)

        self.assertEqual(document_app.load_chunks(stream), [])

    @patch("app.get_client", return_value=object())
    @patch("app.embed", side_effect=RuntimeError("429 RESOURCE_EXHAUSTED"))
    @patch("app.load_chunks", return_value=[{"text": "A passage", "page": 1}])
    def test_upload_shows_plain_rate_limit_message(self, _chunks, _embed, _client):
        response = self.client.post(
            "/api/upload",
            data={"pdf": (BytesIO(b"pdf data"), "sample.pdf")},
            content_type="multipart/form-data",
        )

        self.assertEqual(response.status_code, 429)
        self.assertIn("Please wait a minute", response.json["error"])

    def test_page_exposes_drag_and_drop_upload(self):
        response = self.client.get("/")

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"drag and drop them here", response.data)
        self.assertIn(b"Drop your PDF anywhere to upload", response.data)

    @patch("app.get_client", return_value=object())
    @patch("app.embed", return_value=np.array([[1.0, 0.0]], dtype=np.float32))
    @patch("app.load_chunks", return_value=[{"text": "A passage", "page": 1}])
    def test_upload_indexes_document_and_sanitizes_name(self, _chunks, _embed, _client):
        response = self.client.post(
            "/api/upload",
            data={"pdf": (BytesIO(b"pdf data"), "../../sample document.pdf")},
            content_type="multipart/form-data",
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json["name"], "sample_document.pdf")
        self.assertEqual(response.json["chunks"], 1)
        self.assertTrue(response.json["document_id"])
        document = document_app.documents[response.json["document_id"]]
        self.assertEqual(document["name"], "sample_document.pdf")
        self.assertEqual(document["suggestions"], response.json["suggestions"])
        self.assertTrue(
            (Path(self.data_dir.name) / f"{response.json['document_id']}.json").exists()
        )

    def test_saved_index_loads_without_reembedding(self):
        document = {
            "id": "a" * 32,
            "name": "saved.pdf",
            "chunks": [{"text": "Saved passage", "page": 3}],
            "vectors": np.array([[0.6, 0.8]], dtype=np.float32),
            "suggestions": ["What is saved?"],
        }
        document_app.save_document(document)
        document_app.documents.clear()
        document_app.load_saved_documents()

        self.assertIn(document["id"], document_app.documents)
        np.testing.assert_allclose(
            document_app.documents[document["id"]]["vectors"], [[0.6, 0.8]]
        )

    def test_document_list_includes_passage_counts(self):
        document_app.documents["a" * 32] = {
            "id": "a" * 32,
            "name": "first.pdf",
            "chunks": [{"text": "One", "page": 1}, {"text": "Two", "page": 1}],
            "vectors": np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
            "suggestions": [],
        }

        response = self.client.get("/api/documents")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json["documents"],
            [{"id": "a" * 32, "name": "first.pdf", "chunks": 2}],
        )

    def test_remove_document_deletes_memory_and_saved_index(self):
        document = {
            "id": "b" * 32,
            "name": "remove.pdf",
            "chunks": [{"text": "Passage", "page": 1}],
            "vectors": np.array([[1.0, 0.0]], dtype=np.float32),
            "suggestions": [],
        }
        document_app.documents[document["id"]] = document
        document_app.save_document(document)

        response = self.client.delete(f"/api/documents/{document['id']}")

        self.assertEqual(response.status_code, 200)
        self.assertNotIn(document["id"], document_app.documents)
        self.assertFalse((Path(self.data_dir.name) / f"{document['id']}.json").exists())
        self.assertFalse((Path(self.data_dir.name) / f"{document['id']}.npy").exists())

    def test_remove_document_rejects_invalid_id(self):
        response = self.client.delete("/api/documents/" + "z" * 32)

        self.assertEqual(response.status_code, 400)

    def test_ask_searches_across_documents_and_cites_each_name(self):
        client = Mock()
        client.models.embed_content.return_value = SimpleNamespace(
            embeddings=[SimpleNamespace(values=[1.0, 0.0])]
        )
        client.models.generate_content.return_value = SimpleNamespace(
            text="The answer is in both documents."
        )
        selected_documents = [
            {
                "id": "one",
                "name": "first.pdf",
                "chunks": [{"text": "First source.", "page": 1}],
                "vectors": np.array([[1.0, 0.0]], dtype=np.float32),
                "suggestions": [],
            },
            {
                "id": "two",
                "name": "second.pdf",
                "chunks": [{"text": "Second source.", "page": 4}],
                "vectors": np.array([[0.99, 0.01]], dtype=np.float32),
                "suggestions": [],
            },
        ]

        prompt, sources = document_app.prepare_answer(
            client, "Summarize", selected_documents
        )

        self.assertEqual({source["document"] for source in sources}, {"first.pdf", "second.pdf"})
        self.assertIn("[Document first.pdf, Page 1]", prompt)
        self.assertIn("[Document second.pdf, Page 4]", prompt)

    @patch(
        "app.get_client",
        side_effect=RuntimeError(
            "GEMINI_API_KEY is missing. Add it to the .env file beside app.py."
        ),
    )
    @patch("app.load_chunks", return_value=[{"text": "A passage", "page": 1}])
    def test_upload_reports_missing_api_key(self, _chunks, _client):
        response = self.client.post(
            "/api/upload",
            data={"pdf": (BytesIO(b"pdf data"), "sample.pdf")},
            content_type="multipart/form-data",
        )

        self.assertEqual(response.status_code, 503)
        self.assertIn("GEMINI_API_KEY", response.json["error"])
        self.assertIn(".env file", response.json["error"])
        self.assertEqual(document_app.documents, {})

    def test_ask_requires_json_object(self):
        response = self.client.post("/api/ask", json=["question"])

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json["error"], "Send the question as a JSON object.")

    def test_ask_rejects_question_over_limit(self):
        response = self.client.post(
            "/api/ask",
            json={"question": "x" * (document_app.MAX_QUESTION_CHARS + 1)},
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn("2000 characters", response.json["error"])

    def test_ask_requires_loaded_document(self):
        response = self.client.post("/api/ask", json={"question": "What is this about?"})

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json["error"], "Upload a PDF first.")

    def test_missing_api_key_explains_local_setup(self):
        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, r"GEMINI_API_KEY is missing.*\.env"):
                document_app.get_client()

    def test_suggestions_require_document_id(self):
        response = self.client.post("/api/suggestions", json={})

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json["error"], "A document ID is required.")

    def test_suggestions_require_current_document(self):
        document_app.documents["current-document"] = {
            "id": "current-document",
            "name": "sample.pdf",
            "chunks": [{"text": "A passage", "page": 1}],
            "vectors": np.array([[1.0, 0.0]], dtype=np.float32),
            "suggestions": [],
        }

        response = self.client.post(
            "/api/suggestions", json={"document_id": "different-document"}
        )

        self.assertEqual(response.status_code, 404)

    def test_suggestions_are_built_from_document_topics(self):
        suggestions = document_app.build_question_suggestions(
            [
                {
                    "text": (
                        "Climate policy affects energy and emissions. "
                        "Climate policy can reduce carbon emissions."
                    ),
                    "page": 1,
                },
                {
                    "text": (
                        "Renewable energy lowers carbon emissions. "
                        "Climate policy supports renewable energy."
                    ),
                    "page": 2,
                },
            ]
        )

        self.assertEqual(len(suggestions), 4)
        self.assertEqual(len(set(suggestions)), 4)
        self.assertIn("climate", suggestions[1])

    @patch("app.get_client", side_effect=AssertionError("Suggestions must not call Gemini"))
    def test_suggestions_are_served_from_cached_document_state(self, get_client):
        expected = ["One?", "Two?", "Three?", "Four?"]
        document_app.documents["current-document"] = {
            "id": "current-document",
            "name": "sample.pdf",
            "chunks": [{"text": "A passage", "page": 1}],
            "vectors": np.array([[1.0, 0.0]], dtype=np.float32),
            "suggestions": expected,
        }

        response = self.client.post(
            "/api/suggestions", json={"document_id": "current-document"}
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json["suggestions"], expected)
        get_client.assert_not_called()

    def test_gemini_client_is_reused(self):
        with patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"}):
            with patch.dict(document_app.client_cache, {"api_key": None, "client": None}):
                with patch.object(document_app.genai, "Client") as client_factory:
                    client = client_factory.return_value
                    first = document_app.get_client()
                    second = document_app.get_client()

        self.assertIs(first, client)
        self.assertIs(second, client)
        client_factory.assert_called_once_with(api_key="test-key")

    def test_answer_generation_uses_low_latency_limits(self):
        self.assertEqual(document_app.TOP_K, 2)
        self.assertEqual(document_app.ANSWER_CONFIG.max_output_tokens, 160)
        self.assertEqual(
            document_app.ANSWER_CONFIG.thinking_config.thinking_level,
            document_app.types.ThinkingLevel.MINIMAL,
        )

    @patch("app.get_client")
    def test_ask_returns_answer_and_page_citation(self, get_client):
        client = Mock()
        client.models.embed_content.return_value = SimpleNamespace(
            embeddings=[SimpleNamespace(values=[1.0, 0.0])]
        )
        client.models.generate_content.return_value = SimpleNamespace(
            text="The document says hello."
        )
        get_client.return_value = client
        document_app.documents["sample-id"] = {
            "id": "sample-id",
            "name": "sample.pdf",
            "chunks": [{"text": "Hello from the document.", "page": 2}],
            "vectors": np.array([[1.0, 0.0]], dtype=np.float32),
            "suggestions": [],
        }

        response = self.client.post("/api/ask", json={"question": "What does it say?"})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json["answer"], "The document says hello.")
        self.assertEqual(response.json["sources"][0]["document"], "sample.pdf")
        self.assertEqual(response.json["sources"][0]["page"], 2)
        client.models.generate_content.assert_called_once_with(
            model=document_app.LLM_MODEL,
            contents=unittest.mock.ANY,
            config=document_app.ANSWER_CONFIG,
        )

    @patch("app.get_client")
    def test_ask_streams_answer_text_and_sources(self, get_client):
        client = Mock()
        client.models.embed_content.return_value = SimpleNamespace(
            embeddings=[SimpleNamespace(values=[1.0, 0.0])]
        )
        client.models.generate_content_stream.return_value = iter(
            [SimpleNamespace(text="Hello "), SimpleNamespace(text="world.")]
        )
        get_client.return_value = client
        document_app.documents["sample-id"] = {
            "id": "sample-id",
            "name": "sample.pdf",
            "chunks": [{"text": "Hello from the document.", "page": 2}],
            "vectors": np.array([[1.0, 0.0]], dtype=np.float32),
            "suggestions": [],
        }

        response = self.client.post(
            "/api/ask",
            json={"question": "What does it say?", "stream": True},
            buffered=False,
        )
        first_event = json.loads(next(response.response))
        client.models.embed_content.assert_not_called()
        second_event = json.loads(next(response.response))
        client.models.embed_content.assert_called_once()
        client.models.generate_content_stream.assert_not_called()
        remaining = b"".join(response.response).decode("utf-8")
        response.close()
        events = [first_event, second_event] + [
            json.loads(line) for line in remaining.splitlines()
        ]

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.mimetype, "application/x-ndjson")
        self.assertEqual(first_event["status"], "Finding relevant passages...")
        self.assertEqual(second_event["status"], "Generating answer...")
        self.assertEqual("".join(event.get("text", "") for event in events), "Hello world.")
        self.assertEqual(events[-1]["sources"][0]["document"], "sample.pdf")
        self.assertEqual(events[-1]["sources"][0]["page"], 2)
        client.models.generate_content_stream.assert_called_once_with(
            model=document_app.LLM_MODEL,
            contents=unittest.mock.ANY,
            config=document_app.ANSWER_CONFIG,
        )

    def test_embed_normalizes_vectors(self):
        client = Mock()
        client.models.embed_content.return_value = SimpleNamespace(
            embeddings=[SimpleNamespace(values=[3.0, 4.0])]
        )

        result = document_app.embed(client, ["sample"])

        np.testing.assert_allclose(result, [[0.6, 0.8]])

    def test_embed_rejects_missing_vectors(self):
        client = Mock()
        client.models.embed_content.return_value = SimpleNamespace(embeddings=[])

        with self.assertRaisesRegex(ValueError, "no vectors"):
            document_app.embed(client, ["sample"])


if __name__ == "__main__":
    unittest.main()
