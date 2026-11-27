"""HTTP boundary and progress-stream checks."""

import http.client
import json
import threading
import unittest
from unittest.mock import patch

import serve


class ServerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = serve.create_server(0)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()

    def request(self, body, token=None, origin=None):
        connection = http.client.HTTPConnection(self.server.authority, timeout=5)
        headers = {"Content-Type": "application/json"}
        if token:
            headers["X-Session-Token"] = token
        if origin:
            headers["Origin"] = origin
        connection.request("POST", "/api/classify", json.dumps(body), headers)
        response = connection.getresponse()
        result = response.status, response.read()
        connection.close()
        return result

    def test_requires_session_and_same_origin(self):
        data = {"mode": "text", "text": "Good", "model": "llama3.2:3b"}
        self.assertEqual(self.request(data)[0], 403)
        self.assertEqual(self.request(data, self.server.token, "https://example.com")[0], 403)

    def test_busy_and_bad_csv_have_explicit_errors(self):
        data = {"mode": "csv", "csv": "wrong\nGood", "column": "text", "model": "llama3.2:3b"}
        self.assertEqual(self.request(data, self.server.token)[0], 400)
        self.server.inference_lock.acquire()
        try:
            data = {"mode": "text", "text": "Good", "model": "llama3.2:3b"}
            self.assertEqual(self.request(data, self.server.token)[0], 409)
        finally:
            self.server.inference_lock.release()

    def test_progress_and_download_use_existing_classifier(self):
        identity = {"name": "llama3.2:3b", "digest": "hash", "ollama": "version"}
        data = {"mode": "csv", "csv": '\ufefftext\nGood\n""\n', "column": "text", "model": "llama3.2:3b"}
        with patch.object(serve.classify, "model_identity", return_value=identity), patch.object(
            serve.classify, "predict", return_value="positive"
        ):
            status, content = self.request(data, self.server.token)
        self.assertEqual(status, 200)
        events = [json.loads(line) for line in content.splitlines()]
        self.assertEqual([e["completed"] for e in events if e["type"] == "progress"], [0, 1, 2])
        self.assertEqual(events[-1]["rows"][1]["status"], "empty")
        self.assertIn("model_digest", events[-1]["csv"])

    def test_service_failure_releases_lock_and_returns_error(self):
        with patch.object(serve.classify, "model_identity", side_effect=OSError("offline")):
            status, content = self.request({"mode": "text", "text": "Good", "model": "llama3.2:3b"}, self.server.token)
        self.assertEqual(json.loads(content.splitlines()[-1])["type"], "error")
        self.assertTrue(self.server.inference_lock.acquire(blocking=False))
        self.server.inference_lock.release()


if __name__ == "__main__":
    unittest.main()
