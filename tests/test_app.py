"""Checks for the scientific contract and public API, using the real artifacts."""
import math
import json
from html.parser import HTMLParser
import unittest
from unittest.mock import patch

from app import app, get_predictor, translations
from prediction import CROP_LABELS


class AppTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        app.config["TESTING"] = True
        cls.client = app.test_client()
        cls.base = {"crop": "Boro Rice(BRI 29)", "nk": .12, "p": 12,
                    "area": 10, "area_unit": "decimal"}

    def predict(self, **changes):
        return self.client.post("/api/predict", json={**self.base, **changes})

    def test_pages_and_assets(self):
        for path in ("/", "/predict", "/static/style.css", "/static/app.js",
                     "/static/mark.svg", "/static/translations.json"):
            with self.subTest(path=path):
                response = self.client.get(path)
                self.assertEqual(response.status_code, 200)
                self.assertIn("nosniff", response.headers["X-Content-Type-Options"])
                response.close()

    def test_language_pages_and_default(self):
        with app.test_client() as client:
            page = client.get("/").get_data(as_text=True)
            self.assertIn('<html lang="bn">', page)
            self.assertIn("সারের পরিমাণ হিসাব করুন", page)
            self.assertNotIn("SoilSense", page)
            english = client.get("/predict?lang=en").get_data(as_text=True)
            self.assertIn('<html lang="en">', english)
            self.assertIn("Calculate fertilizer amounts", english)
            client.set_cookie("calculator_language", "en")
            self.assertIn('<html lang="en">', client.get("/").get_data(as_text=True))
            self.assertIn('<html lang="bn">', client.get("/?lang=bn").get_data(as_text=True))
            self.assertIn('<html lang="bn">', client.get("/?lang=invalid").get_data(as_text=True))

    def test_bangla_numerals_match_english_prediction(self):
        bangla = self.predict(nk="০.১২", p="১২", area="১০")
        english = self.predict()
        self.assertEqual(bangla.status_code, 200)
        self.assertEqual(bangla.json["fertilizers"], english.json["fertilizers"])
        self.assertEqual(self.predict(nk="৫০").json["code"], "invalid_range")

    def test_translation_keys_and_template_bindings(self):
        copies = translations()
        self.assertEqual(copies["bn"].keys(), copies["en"].keys())
        for locale in copies.values():
            self.assertEqual(set(locale["crops"]), set(CROP_LABELS))
            for key, value in locale.items():
                self.assertTrue(value, key)

        class Bindings(HTMLParser):
            def __init__(self):
                super().__init__()
                self.keys = []
                self.embedded = None

            def handle_starttag(self, tag, attrs):
                attrs = dict(attrs)
                for attr in ("data-i18n", "data-i18n-placeholder", "data-i18n-aria", "data-submit-label", "data-label"):
                    if attr in attrs:
                        self.keys.append(attrs[attr])
                if attrs.get("id") == "app-copy":
                    self.embedded = json.loads(attrs["data-translations"])

        parser = Bindings()
        parser.feed(self.client.get("/").get_data(as_text=True))
        for key in parser.keys:
            self.assertIn(key, copies["bn"])
            self.assertIn(key, copies["en"])
        self.assertEqual(parser.embedded, copies)

    def test_language_independent_errors(self):
        result = self.predict(nk=50).json
        self.assertEqual(result["code"], "invalid_range")
        self.assertEqual(result["params"], {"min": 0, "max": .36})
        self.assertEqual(self.predict(crop="wheat").json["code"], "invalid_crop")
        self.assertEqual(self.predict(area="না").json["code"], "invalid_number")

    def test_known_model_prediction_and_target_order(self):
        data = self.predict().get_json()
        self.assertEqual([x["name"] for x in data["fertilizers"]], ["Urea", "MoP", "TSP"])
        # Recorded directly from the supplied model at the paired guide example.
        for result, expected in zip(data["fertilizers"], [1390.89, 729.0833333333334, 317.9266666666667]):
            self.assertAlmostEqual(result["rate_g_per_decimal"], expected, places=7)
            self.assertAlmostEqual(result["total_g"], expected * 10)
            self.assertAlmostEqual(result["total_kg"], expected / 100)
        self.assertTrue(data["on_training_diagonal"])
        self.assertEqual(data["warnings"], [])

    def test_all_supported_crops_and_boundary_values(self):
        for crop in CROP_LABELS:
            for nk, p in [(0, 0), (.36, 36)]:
                with self.subTest(crop=crop, nk=nk):
                    response = self.predict(crop=crop, nk=nk, p=p)
                    self.assertEqual(response.status_code, 200)
                    for item in response.json["fertilizers"]:
                        self.assertTrue(math.isfinite(item["total_g"]))
                        self.assertGreaterEqual(item["total_g"], 0)

    def test_equivalent_area_units(self):
        decimal = self.predict(area=100).json
        acre = self.predict(area=1, area_unit="acre").json
        hectare = self.predict(area=.40468564224, area_unit="hectare").json
        for alternative in (acre, hectare):
            for x, y in zip(decimal["fertilizers"], alternative["fertilizers"]):
                self.assertAlmostEqual(x["total_g"], y["total_g"], places=6)

    def test_off_diagonal_warning(self):
        data = self.predict(nk=.12, p=20).json
        self.assertFalse(data["on_training_diagonal"])
        self.assertIn("exploratory", data["warnings"][0])

    def test_rejects_invalid_input(self):
        invalid = [("nk", 50), ("nk", -.1), ("p", 37), ("p", "NaN"),
                   ("area", 0), ("area", -1), ("area", "Infinity"), ("area", 1e-10),
                   ("area", 1e7), ("nk", True), ("nk", []), ("p", {}),
                   ("crop", "wheat"), ("crop", []), ("area_unit", "bigha")]
        for field, value in invalid:
            with self.subTest(field=field, value=value):
                response = self.predict(**{field: value})
                self.assertEqual(response.status_code, 400)
                self.assertEqual(response.json["field"], field)

    def test_missing_and_malformed_payloads(self):
        for payload in (None, [], {}, "string"):
            response = self.client.post("/api/predict", json=payload)
            self.assertEqual(response.status_code, 400)
        response = self.client.post("/api/predict", data="{bad", content_type="application/json")
        self.assertEqual(response.status_code, 400)

    def test_calibration_equations(self):
        response = self.client.post("/api/calibrate", json={"n": .003, "p": 104, "k": .248})
        self.assertEqual(response.status_code, 200)
        values = response.json["readings"]
        for value, expected in zip(values, [.0527365, 22.62144, .395632]):
            self.assertAlmostEqual(value["corrected"], expected, places=7)
        self.assertEqual(values[1]["quality"], "Not quantitatively supported")
        self.assertIn("not passed", response.json["limitation"])

    def test_calibration_range_enforced(self):
        for key, bad in [("n", 0), ("p", 71), ("k", .9)]:
            data = {"n": .003, "p": 104, "k": .248, key: bad}
            response = self.client.post("/api/calibrate", json=data)
            self.assertEqual(response.status_code, 400)
            self.assertEqual(response.json["field"], key)

    def test_model_unavailable_is_explained_without_leaking_exception(self):
        with patch("app.get_predictor", side_effect=RuntimeError("private filesystem detail")):
            with self.assertLogs(app.logger, level="ERROR"):
                response = self.predict()
            self.assertEqual(response.status_code, 503)
            self.assertNotIn("private filesystem", response.get_data(as_text=True))
        self.assertEqual(len(get_predictor().crops), 9)


if __name__ == "__main__":
    unittest.main()
