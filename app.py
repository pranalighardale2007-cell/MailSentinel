import io
import ipaddress
import re
import sqlite3
import uuid
from datetime import datetime
from urllib.parse import parse_qs, unquote, urlparse

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


st.set_page_config(
	page_title="MailSentry | Fake Email Detection",
	page_icon="🛡️",
	layout="wide",
	initial_sidebar_state="expanded",
)


SUSPICIOUS_WORDS = {
	"urgent": 10,
	"immediately": 10,
	"verify": 8,
	"suspended": 12,
	"expire": 8,
	"congratulations": 9,
	"winner": 9,
	"claim": 7,
	"refund": 7,
	"act now": 12,
	"limited time": 8,
}

SCAM_PHRASES = [
	"your account will be closed",
	"click here to verify",
	"you have been selected",
	"unusual activity",
	"failure to respond",
	"kindly send",
	"keep this confidential",
	"you must act",
]

TRUSTED_LINK_DOMAINS = {
	"amazon.com", "apple.com", "dropbox.com", "facebook.com", "google.com",
	"github.com", "linkedin.com", "microsoft.com", "paypal.com", "theguardian.com",
	"twitter.com", "x.com", "youtube.com", "zoom.us",
}
URL_PHISHING_KEYWORDS = {
	"account", "authenticate", "confirm", "credential", "login", "password",
	"recover", "secure", "signin", "unlock", "update", "verify", "wallet",
}
URL_SHORTENERS = {"bit.ly", "t.co", "tinyurl.com", "ow.ly", "is.gd", "cutt.ly", "rb.gy"}

TRAINING_TEXT = [
	"team meeting moved to tomorrow morning",
	"please find the project notes attached",
	"your package delivery is scheduled for Friday",
	"thanks for sending the quarterly report",
	"can we reschedule our appointment",
	"your invoice is available in the customer portal",
	"urgent verify your account immediately click here",
	"your password expires today confirm your login",
	"you won congratulations claim your prize now",
	"bank account suspended send your otp and password",
	"click this link to receive your refund",
	"unusual activity confirm your credit card details",
]
TRAINING_LABELS = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]

DB_PATH = "mailsentry_reports.db"
TRACKING_STEPS = [
	("Submitted", "✅", "Report received by the simulated cybersecurity team."),
	("Screening", "✅", "Initial content and risk screening completed."),
	("Investigation", "🔵", "A specialist is reviewing the reported indicators."),
	("Geo/IP Analysis", "⏳", "Sender infrastructure will be checked next."),
	("Threat Assessment", "⏳", "The team will assess the overall threat level."),
	("Action Taken", "⏳", "Recommended protective action will be recorded."),
	("Resolved", "⏳", "The simulated case will be closed."),
]


def get_database_connection():
	connection = sqlite3.connect(DB_PATH)
	connection.row_factory = sqlite3.Row
	return connection


def initialize_database():
	with get_database_connection() as connection:
		connection.execute(
			"""
			CREATE TABLE IF NOT EXISTS reports (
				report_id TEXT PRIMARY KEY,
				sender TEXT NOT NULL,
				subject TEXT,
				verdict TEXT NOT NULL,
				risk_score INTEGER NOT NULL,
				report_text TEXT NOT NULL,
				status_index INTEGER NOT NULL DEFAULT 0,
				created_at TEXT NOT NULL,
				updated_at TEXT NOT NULL
			)
			"""
		)
		connection.execute(
			"""
			CREATE TABLE IF NOT EXISTS report_activity (
				activity_id INTEGER PRIMARY KEY AUTOINCREMENT,
				report_id TEXT NOT NULL,
				stage TEXT NOT NULL,
				message TEXT NOT NULL,
				created_at TEXT NOT NULL,
				FOREIGN KEY (report_id) REFERENCES reports (report_id)
			)
			"""
		)
		for column, definition in {
			"priority": "TEXT NOT NULL DEFAULT 'Medium'",
			"case_status": "TEXT NOT NULL DEFAULT 'Open'",
			"assignee": "TEXT NOT NULL DEFAULT 'Unassigned'",
			"tags": "TEXT NOT NULL DEFAULT ''",
			"analyst_notes": "TEXT NOT NULL DEFAULT ''",
			"containment_action": "TEXT NOT NULL DEFAULT ''",
		}.items():
			try:
				connection.execute(f"ALTER TABLE reports ADD COLUMN {column} {definition}")
			except sqlite3.OperationalError as error:
				if "duplicate column name" not in str(error).lower():
					raise


def extract_evidence(sender, subject, body):
	combined = f"{subject} {body}".strip().lower()
	links = re.findall(r"https?://[^\s<>()]+|www\.[^\s<>()]+", combined)
	domain = extract_domain(sender)
	ip_addresses = re.findall(r"\b(?:\d{1,3}\.){3}\d{1,3}\b", combined)
	hashes = re.findall(r"\b[a-f0-9]{32,64}\b", combined)
	return {
		"sender": sender.strip() or "Unavailable",
		"domain": domain or "Unavailable",
		"sender_format": "Valid email format" if re.fullmatch(r"[^\s@]+@[^\s@]+\.[^\s@]+", sender.strip()) else "Unusual or unavailable",
		"links": links or ["Unavailable"],
		"ip_address": ", ".join(dict.fromkeys(ip_addresses)) or "Unavailable",
		"geolocation": "Unavailable",
		"hashes": list(dict.fromkeys(hashes)),
		"evidence_note": "No IP or geolocation evidence was supplied in this email.",
	}


def create_report_text(sender, subject, body, result):
	created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
	reasons = "\n".join(f"- {reason}" for reason in result["reasons"])
	evidence = extract_evidence(sender, subject, body)
	links = "\n".join(f"- {link}" for link in evidence["links"])
	hashes = "\n".join(f"- {item}" for item in evidence["hashes"]) or "- None found"
	return (
		"MAILSENTRY CYBERSECURITY INVESTIGATION REPORT\n"
		+ "=" * 36
		+ f"\nGenerated: {created_at}\nSender: {sender or 'Not provided'}\n"
		+ f"Subject: {subject or 'Not provided'}\nVerdict: {result['verdict']}\n"
		+ f"Risk score: {result['risk_score']}%\nPriority: {result['priority']}\n\nSender details:\n"
		+ f"- Domain: {evidence['domain']}\n- Format: {evidence['sender_format']}\n"
		+ f"- IP address: {evidence['ip_address']}\n- Geolocation: {evidence['geolocation']}\n\n"
		+ f"Suspicious links:\n{links}\n\nFile hashes:\n{hashes}\n\nDetection reasons:\n{reasons}\n\n"
		+ f"Recommendation: {result['recommendation']}\n\nAnalyzed email body:\n{body or 'Not provided'}"
	)


def make_pdf(report_text):
	"""Create a small text PDF without requiring an additional PDF package."""
	lines = report_text.splitlines() or ["MailSentry report"]
	wrapped_lines = []
	for line in lines:
		while len(line) > 92:
			wrapped_lines.append(line[:92])
			line = line[92:]
		wrapped_lines.append(line)
	pages = [wrapped_lines[index:index + 48] for index in range(0, len(wrapped_lines), 48)] or [[]]
	objects = [None, b"<< /Type /Catalog /Pages 2 0 R >>", None, b"<< /Type /Font /Subtype /Type1 /BaseFont /Courier >>"]
	page_references = []
	for page_lines in pages:
		content_lines = ["BT", "/F1 9 Tf", "50 750 Td", "12 TL"]
		for line in page_lines:
			safe_line = line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
			content_lines.append(f"({safe_line.encode('latin-1', 'replace').decode('latin-1')}) Tj T*")
		content = "\n".join(content_lines + ["ET"]).encode("latin-1", "replace")
		content_object = len(objects)
		objects.append(f"<< /Length {len(content)} >>\nstream\n".encode() + content + b"\nendstream")
		page_object = len(objects)
		objects.append(None)
		page_references.append(page_object)
		objects[page_object] = f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 4 0 R >> >> /Contents {content_object} 0 R >>".encode()
	objects[2] = ("<< /Type /Pages /Kids [" + " ".join(f"{ref} 0 R" for ref in page_references) + f"] /Count {len(page_references)} >>").encode()
	pdf = bytearray(b"%PDF-1.4\n")
	offsets = [0]
	for object_number, obj in enumerate(objects[1:], start=1):
		offsets.append(len(pdf))
		pdf.extend(f"{object_number} 0 obj\n".encode())
		pdf.extend(obj)
		pdf.extend(b"\nendobj\n")
	xref_position = len(pdf)
	pdf.extend(f"xref\n0 {len(objects)}\n0000000000 65535 f \n".encode())
	for offset in offsets[1:]:
		pdf.extend(f"{offset:010d} 00000 n \n".encode())
	pdf.extend(f"trailer\n<< /Size {len(objects)} /Root 1 0 R >>\nstartxref\n{xref_position}\n%%EOF".encode())
	return bytes(pdf)


def save_report(sender, subject, report_text, result):
	report_id = f"MS-{datetime.now():%Y%m%d}-{uuid.uuid4().hex[:8].upper()}"
	now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
	with get_database_connection() as connection:
		connection.execute(
			"INSERT INTO reports (report_id, sender, subject, verdict, risk_score, report_text, status_index, created_at, updated_at, priority) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
			(report_id, sender, subject, result["verdict"], result["risk_score"], report_text, 2, now, now, result["priority"]),
		)
		connection.execute(
			"INSERT INTO report_activity (report_id, stage, message, created_at) VALUES (?, ?, ?, ?)",
			(report_id, "Investigation", "Report submitted for simulated cybersecurity review.", now),
		)
	return report_id


def get_report(report_id):
	with get_database_connection() as connection:
		return connection.execute("SELECT * FROM reports WHERE report_id = ?", (report_id.strip().upper(),)).fetchone()


def get_report_activity(report_id):
	with get_database_connection() as connection:
		return connection.execute(
			"SELECT stage, message, created_at FROM report_activity WHERE report_id = ? ORDER BY activity_id DESC",
			(report_id,),
		).fetchall()


def get_dashboard_reports():
	with get_database_connection() as connection:
		return pd.read_sql_query("SELECT report_id, sender, subject, verdict, risk_score, status_index, priority, case_status, assignee, tags, analyst_notes, containment_action, created_at, updated_at FROM reports ORDER BY updated_at DESC", connection)


def update_case(report_id, priority, case_status, assignee, tags, analyst_notes, containment_action):
	now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
	with get_database_connection() as connection:
		connection.execute(
			"UPDATE reports SET priority = ?, case_status = ?, assignee = ?, tags = ?, analyst_notes = ?, containment_action = ?, updated_at = ? WHERE report_id = ?",
			(priority, case_status, assignee.strip() or "Unassigned", tags.strip(), analyst_notes.strip(), containment_action.strip(), now, report_id),
		)
		connection.execute(
			"INSERT INTO report_activity (report_id, stage, message, created_at) VALUES (?, ?, ?, ?)",
			(report_id, "Analyst update", f"Case updated: {case_status} / {priority} / {assignee.strip() or 'Unassigned'}.", now),
		)


initialize_database()


def build_model():
	"""Create a tiny local baseline model; no download or API is needed."""
	model = Pipeline(
		[
			("vectorizer", TfidfVectorizer(ngram_range=(1, 2), lowercase=True)),
			("classifier", LogisticRegression(random_state=42, max_iter=1000)),
		]
	)
	model.fit(TRAINING_TEXT, TRAINING_LABELS)

	# Exercise joblib in memory so the app has a serializable model without files.
	buffer = io.BytesIO()
	joblib.dump(model, buffer)
	buffer.seek(0)
	return joblib.load(buffer)


@st.cache_resource
def get_model():
	return build_model()


def extract_domain(sender):
	match = re.search(r"@([a-z0-9.-]+)", sender.lower())
	return match.group(1) if match else ""


def classify_url(url, sender_domain=""):
	clean_url = url.rstrip(".,;:!?)]}")
	parsed = urlparse(clean_url if re.match(r"^[a-z][a-z0-9+.-]*://", clean_url, re.IGNORECASE) else f"http://{clean_url}")
	host = (parsed.hostname or "").lower().rstrip(".")
	base_domain = host[4:] if host.startswith("www.") else host
	url_text = unquote(clean_url).lower()
	reasons = []

	if not host:
		return {"url": url, "label": "Suspicious Link", "suspicious": True, "reasons": ["The URL has no valid domain."], "score": 10}

	try:
		ipaddress.ip_address(host)
		is_ip_host = True
	except ValueError:
		is_ip_host = False

	trusted = any(base_domain == domain or base_domain.endswith(f".{domain}") for domain in TRUSTED_LINK_DOMAINS)
	if is_ip_host:
		reasons.append("Uses an IP address instead of a normal domain.")
	if base_domain in URL_SHORTENERS:
		reasons.append("Uses a URL-shortening service that hides the destination.")
	if not trusted and any(keyword in url_text for keyword in URL_PHISHING_KEYWORDS):
		reasons.append("Contains a phishing-related keyword in the destination.")
	if not trusted and ("xn--" in base_domain or base_domain.count("-") >= 3 or any(len(part) >= 24 for part in base_domain.split("."))):
		reasons.append("The domain has an unusual or obfuscated structure.")
	if not trusted and (len(re.findall(r"%[0-9a-f]{2}", clean_url.lower())) >= 3 or len(re.findall(r"https?%3a|https?://", clean_url.lower())) > 1):
		reasons.append("Contains encoded or nested redirect data.")
	query = parse_qs(parsed.query)
	if not trusted and any(key.lower() in {"url", "u", "target", "redirect", "redirect_url", "next", "continue", "dest", "destination"} for key in query):
		reasons.append("Contains a redirect parameter that can conceal the final destination.")

	known_brands = {domain.split(".")[0] for domain in TRUSTED_LINK_DOMAINS if len(domain.split(".")[0]) >= 4}
	if not trusted:
		host_labels = base_domain.split(".")
		for brand in known_brands:
			if any(label == brand or label.startswith(f"{brand}-") for label in host_labels):
				reasons.append(f"Looks like an impersonation of the {brand} domain.")
				break
	if sender_domain and not trusted:
		sender_base = sender_domain[4:] if sender_domain.startswith("www.") else sender_domain
		brand = sender_base.split(".")[0]
		if len(brand) >= 4 and brand in base_domain and not (base_domain == sender_base or base_domain.endswith(f".{sender_base}")):
			reasons.append("The destination imitates the sender's domain without matching it.")

	suspicious = bool(reasons)
	return {
		"url": url,
		"label": "Suspicious Link" if suspicious else "Safe/Normal Link",
		"suspicious": suspicious,
		"reasons": reasons,
		"score": min(24, len(reasons) * 6) if suspicious else 0,
	}


def analyze_email(sender, subject, body):
	combined = f"{subject} {body}".strip().lower()
	reasons = []
	signals = []
	score = 0

	model = get_model()
	model_probability = float(model.predict_proba([combined])[0][1])
	score += model_probability * 35

	if not sender or not re.fullmatch(r"[^\s@]+@[^\s@]+\.[^\s@]+", sender.strip()):
		score += 18
		signals.append("Sender format looks incomplete or unusual.")

	domain = extract_domain(sender)
	if domain and any(token in domain for token in ("gmail-secure", "verify", "account", "support-", "login-")):
		score += 15
		signals.append(f"The sender domain '{domain}' contains a deceptive-looking keyword.")

	found_words = []
	for word, points in SUSPICIOUS_WORDS.items():
		if re.search(rf"\b{re.escape(word)}\b", combined):
			found_words.append(word)
			score += points
	if found_words:
		shown_words = ", ".join(found_words[:5])
		suffix = "..." if len(found_words) > 5 else ""
		signals.append(f"Suspicious wording detected: {shown_words}{suffix}.")

	links = re.findall(r"https?://[^\s<>()]+|www\.[^\s<>()]+", combined)
	link_results = [classify_url(link, domain) for link in links]
	suspicious_links = [item for item in link_results if item["suspicious"]]
	if suspicious_links:
		score += sum(item["score"] for item in suspicious_links)
		signals.append(f"Suspicious link indicator detected in {len(suspicious_links)} of {len(links)} link{'s' if len(links) != 1 else ''}.")
		for item in suspicious_links[:3]:
			signals.extend(item["reasons"][:2])

	request_checks = {
		"OTP or verification code": r"\b(otp|one[- ]time password|verification code|security code)\b",
		"Password or login details": r"\b(password|passcode|login details|username)\b",
		"Bank or card information": r"\b(bank|account number|credit card|debit card|routing number|cvv)\b",
		"Money or payment": r"\b(payment|pay|transfer|gift card|wire|fee)\b",
	}
	requests = [label for label, pattern in request_checks.items() if re.search(pattern, combined)]
	if requests:
		score += min(32, len(requests) * 10)
		signals.append("Requests sensitive information: " + ", ".join(requests) + ".")

	matched_phrases = [phrase for phrase in SCAM_PHRASES if phrase in combined]
	if matched_phrases:
		score += min(20, len(matched_phrases) * 7)
		signals.append("Common scam language detected: " + ", ".join(matched_phrases[:3]) + ".")

	if re.search(r"[A-Z]{4,}|!{2,}|\$\s?\d+", f"{subject} {body}"):
		score += 5
		signals.append("The message uses attention-grabbing formatting or money amounts.")

	risk_score = int(np.clip(round(score), 0, 100))
	is_fake = risk_score >= 50
	priority = "Critical" if risk_score >= 80 else "High" if risk_score >= 65 else "Medium" if risk_score >= 35 else "Low"
	if not signals:
		signals.append("No strong phishing indicators were found in the supplied content.")
	if is_fake:
		verdict = "FAKE / SUSPICIOUS"
		recommendation = "Do not click links or share information. Contact the sender through a trusted channel."
	else:
		verdict = "LIKELY SAFE"
		recommendation = "The email looks low-risk, but still check the sender and links before taking action."

	return {
		"verdict": verdict,
		"is_fake": is_fake,
		"risk_score": risk_score,
		"priority": priority,
		"reasons": signals,
		"recommendation": recommendation,
		"link_results": link_results,
		"metrics": {
			"Suspicious words": len(found_words),
			"Links found": len(links),
			"Sensitive requests": len(requests),
			"Scam phrases": len(matched_phrases),
			"Indicators": len(links) + len(re.findall(r"\b(?:\d{1,3}\.){3}\d{1,3}\b|\b[a-f0-9]{32,64}\b", combined)),
		},
	}


def render_gauge(score):
	color = "#ff5c70" if score >= 50 else "#47e6a1"
	figure = go.Figure(
		go.Indicator(
			mode="gauge+number",
			value=score,
			number={"suffix": "%", "font": {"size": 42, "color": "#f3f7ff"}},
			title={"text": "RISK SCORE", "font": {"size": 14, "color": "#8fa2bd"}},
			gauge={
				"axis": {"range": [0, 100], "tickcolor": "#52627a", "dtick": 25},
				"bar": {"color": color, "thickness": 0.25},
				"bgcolor": "#182338",
				"borderwidth": 0,
				"steps": [
					{"range": [0, 35], "color": "#173d3e"},
					{"range": [35, 65], "color": "#4b3f22"},
					{"range": [65, 100], "color": "#482936"},
				],
			},
		)
	)
	figure.update_layout(height=260, margin={"t": 35, "b": 10, "l": 25, "r": 25}, paper_bgcolor="rgba(0,0,0,0)")
	st.plotly_chart(figure, use_container_width=True, config={"displayModeBar": False})


st.markdown(
	"""
	<style>
	@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
	:root { --bg: #0a1020; --panel: #111a2d; --line: #24324b; --text: #f3f7ff; --muted: #8fa2bd; --cyan: #5de1ff; --green: #47e6a1; --red: #ff5c70; }
	.stApp { background: radial-gradient(circle at 82% 0%, #182c4a 0, var(--bg) 38%); color: var(--text); font-family: 'DM Sans', sans-serif; }
	h1, h2, h3 { font-family: 'Space Grotesk', sans-serif; }
	.block-container { max-width: 1240px; padding-top: 2.3rem; }
	[data-testid='stSidebar'] { background: #0d1628; border-right: 1px solid var(--line); }
	[data-testid='stSidebar'] * { color: var(--muted); }
	.brand { color: var(--text); font: 700 1.35rem 'Space Grotesk'; letter-spacing: -.03em; margin: 0 0 2.5rem; }
	.brand span { color: var(--cyan); }
	.eyebrow { color: var(--cyan); font-size: .76rem; font-weight: 700; letter-spacing: .16em; text-transform: uppercase; }
	.hero { border-bottom: 1px solid var(--line); margin-bottom: 1.5rem; padding-bottom: 1.2rem; }
	.hero h1 { font-size: clamp(2rem, 4vw, 3.5rem); letter-spacing: -.055em; margin: .35rem 0 .5rem; }
	.hero p { color: var(--muted); font-size: 1.02rem; margin: 0; max-width: 670px; }
	.panel { background: rgba(17, 26, 45, .86); border: 1px solid var(--line); border-radius: 12px; padding: 1.35rem; }
	.panel-title { color: var(--text); font: 600 1rem 'Space Grotesk'; margin-bottom: 1rem; }
	.hint { color: var(--muted); font-size: .82rem; }
	div[data-testid='stTextInput'] label, div[data-testid='stTextArea'] label { color: #dce7f5; font-weight: 600; }
	div[data-testid='stTextInput'] input, div[data-testid='stTextArea'] textarea { background: #0b1425; border: 1px solid var(--line); color: var(--text); border-radius: 8px; }
	div.stButton > button { background: var(--cyan); border: 0; border-radius: 8px; color: #071321; font-weight: 700; min-height: 2.8rem; width: 100%; }
	div.stButton > button:hover { background: #a4efff; color: #071321; }
	.result { border-radius: 10px; padding: 1rem; margin-bottom: 1rem; }
	.result.fake { background: rgba(255, 92, 112, .1); border: 1px solid rgba(255, 92, 112, .45); }
	.result.safe { background: rgba(71, 230, 161, .1); border: 1px solid rgba(71, 230, 161, .45); }
	.result strong { font: 700 1.25rem 'Space Grotesk'; }
	.result.fake strong { color: var(--red); } .result.safe strong { color: var(--green); }
	.reason { border-bottom: 1px solid var(--line); color: #c7d4e6; padding: .75rem 0; }
	.reason:last-child { border-bottom: 0; }
	.report-box { background: #0b1425; border: 1px solid var(--line); border-radius: 10px; padding: 1rem; margin-top: 1rem; }
	.report-id { color: var(--cyan); font: 700 1.1rem 'Space Grotesk'; letter-spacing: .04em; }
	.tracker-step { border-left: 2px solid var(--line); margin-left: .7rem; padding: .25rem 0 .8rem 1rem; position: relative; }
	.tracker-step::before { background: #52627a; border: 3px solid #0d1628; border-radius: 50%; content: ''; height: .8rem; left: -.52rem; position: absolute; top: .35rem; width: .8rem; }
	.tracker-step.done { border-left-color: var(--green); } .tracker-step.done::before { background: var(--green); }
	.tracker-step.current { border-left-color: var(--cyan); } .tracker-step.current::before { background: var(--cyan); }
	.tracker-label { color: var(--text); font-weight: 700; } .tracker-note { color: var(--muted); font-size: .82rem; }
	.evidence-card { background: #0b1425; border: 1px solid var(--line); border-radius: 10px; padding: .9rem; min-height: 5.2rem; }
	.evidence-label { color: var(--muted); font-size: .72rem; letter-spacing: .08em; text-transform: uppercase; }
	.evidence-value { color: var(--text); font-weight: 700; margin-top: .35rem; word-break: break-word; }
	.activity-row { border-bottom: 1px solid var(--line); padding: .65rem 0; }
	.activity-row:last-child { border-bottom: 0; }
	footer { color: #52627a; font-size: .8rem; margin-top: 2rem; text-align: center; }
	</style>
	""",
	unsafe_allow_html=True,
)


with st.sidebar:
	st.markdown("<div class='brand'>MAIL<span>SENTRY</span></div>", unsafe_allow_html=True)
	st.markdown("### How it works")
	st.caption("MailSentry checks message language, sender patterns, links, and sensitive information requests using a local analysis model.")
	st.markdown("### Privacy")
	st.caption("Analysis stays local. Reports are stored only in this app's local SQLite file; nothing is sent to an external service.")
	st.markdown("### Risk guide")
	st.markdown("🟢 **0–34%**  Low risk\n\n🟡 **35–64%**  Review carefully\n\n🔴 **65–100%**  High risk")


st.markdown(
	"<div class='hero'><div class='eyebrow'>Local threat analysis · No data leaves your browser</div><h1>MailSentinel</h1><p>Analyze suspicious emails for phishing language, risky requests, and deceptive links before they become a problem.</p></div>",
	unsafe_allow_html=True,
)

input_col, result_col = st.columns([1.08, 0.92], gap="large")
with input_col:
	st.markdown("<div class='panel-title'>EMAIL CONTENT</div>", unsafe_allow_html=True)
	sender = st.text_input("Sender email", placeholder="security@example.com")
	subject = st.text_input("Subject", placeholder="Your account needs attention")
	body = st.text_area("Email body", height=250, placeholder="Paste the full email message here...")
	analyze = st.button("🛡️  Analyze Email")
	st.markdown("<p class='hint'>For the clearest result, include the complete sender address and message body.</p>", unsafe_allow_html=True)

if "analysis_result" not in st.session_state:
	st.session_state.analysis_result = None
if "report_text" not in st.session_state:
	st.session_state.report_text = None
if "report_id" not in st.session_state:
	st.session_state.report_id = None

if analyze:
	if not sender.strip() and not subject.strip() and not body.strip():
		st.warning("Add an email address, subject, or body before analyzing.")
	else:
		st.session_state.analysis_result = analyze_email(sender, subject, body)
		st.session_state.report_text = None
		st.session_state.report_id = None

with result_col:
	st.markdown("<div class='panel-title'>ANALYSIS RESULT</div>", unsafe_allow_html=True)
	result = st.session_state.analysis_result
	if result:
			result_class = "fake" if result["is_fake"] else "safe"
			icon = "⚠️" if result["is_fake"] else "✓"
			st.markdown(
				f"<div class='result {result_class}'><strong>{icon} {result['verdict']}</strong><br><span class='hint'>{result['recommendation']}</span></div>",
				unsafe_allow_html=True,
			)
			render_gauge(result["risk_score"])
			metric_cols = st.columns(5)
			for column, (label, value) in zip(metric_cols, result["metrics"].items()):
				column.metric(label, value)
			st.markdown("### Why this result")
			for reason in result["reasons"]:
				st.markdown(f"<div class='reason'>• {reason}</div>", unsafe_allow_html=True)
			evidence = extract_evidence(sender, subject, body)
			st.markdown("### Evidence collected", unsafe_allow_html=True)
			evidence_cols = st.columns(4)
			for column, label, value in zip(
				evidence_cols,
				["Sender domain", "Sender format", "IP address", "Geolocation"],
				[evidence["domain"], evidence["sender_format"], evidence["ip_address"], evidence["geolocation"]],
			):
				column.markdown(f"<div class='evidence-card'><div class='evidence-label'>{label}</div><div class='evidence-value'>{value}</div></div>", unsafe_allow_html=True)
			st.markdown("**Link analysis**")
			if result["link_results"]:
				for link_result in result["link_results"]:
					label = link_result["label"]
					marker = "⚠️" if link_result["suspicious"] else "✅"
					st.markdown(f"- {marker} **{label}**: `{link_result['url']}`")
					for reason in link_result["reasons"]:
						st.caption(reason)
			else:
				st.caption("No links detected.")
			st.markdown("### Cybersecurity report", unsafe_allow_html=True)
			if st.button("📄  Generate Cybersecurity Report"):
				st.session_state.report_text = create_report_text(sender, subject, body, result)
				st.session_state.report_id = None
			if st.session_state.report_text:
				st.download_button(
					"⬇️  Download Report as PDF",
					data=make_pdf(st.session_state.report_text),
					file_name="mailsentry_cybersecurity_report.pdf",
					mime="application/pdf",
					use_container_width=True,
				)
				st.markdown("<div class='report-box'><span class='hint'>Report ready for local simulated submission.</span></div>", unsafe_allow_html=True)
				if st.session_state.report_id:
					st.markdown(f"<div class='report-box'>Report ID<br><span class='report-id'>{st.session_state.report_id}</span></div>", unsafe_allow_html=True)
				else:
					if st.button("📤  Submit Report"):
						st.session_state.report_id = save_report(sender, subject, st.session_state.report_text, result)
						st.session_state.tracked_report_id = st.session_state.report_id
						st.success("Report submitted to the simulated cybersecurity team.")
	else:
		st.info("Your analysis will appear here after you submit an email.")

st.markdown("---")
st.markdown("## Track Report")
st.caption("Enter a Report ID to view its simulated cybersecurity-team progress. No report is transmitted anywhere.")
track_col, track_button_col = st.columns([1, .28])
with track_col:
	track_id = st.text_input("Report ID", placeholder="MS-20260905-XXXXXXXX", label_visibility="collapsed")
with track_button_col:
	track = st.button("🔎  Track Report")

if track:
	tracked_report = get_report(track_id)
	if tracked_report is None:
		st.error("Report ID not found in the local report store.")
	else:
		st.session_state.tracked_report_id = tracked_report["report_id"]

tracked_report_id = st.session_state.get("tracked_report_id")
if tracked_report_id:
	tracked_report = get_report(tracked_report_id)
	if tracked_report:
		current_index = tracked_report["status_index"]
		current_name = TRACKING_STEPS[current_index][0]
		progress = int(round((current_index + 1) / len(TRACKING_STEPS) * 100))
		st.markdown(f"<div class='report-box'><span class='hint'>Tracking</span><br><span class='report-id'>{tracked_report['report_id']}</span></div>", unsafe_allow_html=True)
		st.metric("Current status", current_name, f"{progress}% complete")
		st.progress(progress, text=f"Cybersecurity workflow: {progress}%")
		for index, (name, default_icon, note) in enumerate(TRACKING_STEPS):
			if index < current_index:
				icon, state_class = "✅", "done"
			elif index == current_index:
				icon, state_class = "🔵", "current"
			else:
				icon, state_class = "⏳", ""
			st.markdown(f"<div class='tracker-step {state_class}'><span class='tracker-label'>{icon} {name}</span><br><span class='tracker-note'>{note}</span></div>", unsafe_allow_html=True)
		st.caption(f"Last update: {tracked_report['updated_at']} · Next step: {TRACKING_STEPS[min(current_index + 1, len(TRACKING_STEPS) - 1)][0]}")
		st.markdown("### Investigation activity")
		for activity in get_report_activity(tracked_report["report_id"]):
			st.markdown(f"<div class='activity-row'><strong>{activity['stage']}</strong><br><span class='hint'>{activity['message']} · {activity['created_at']}</span></div>", unsafe_allow_html=True)
		if current_index < len(TRACKING_STEPS) - 1:
			if st.button("⏩  Advance Simulated Status"):
				now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
				next_stage = TRACKING_STEPS[current_index + 1][0]
				with get_database_connection() as connection:
					connection.execute("UPDATE reports SET status_index = ?, updated_at = ? WHERE report_id = ?", (current_index + 1, now, tracked_report["report_id"]))
					connection.execute("INSERT INTO report_activity (report_id, stage, message, created_at) VALUES (?, ?, ?, ?)", (tracked_report["report_id"], next_stage, f"Simulated team advanced the case to {next_stage}.", now))
				st.rerun()

st.markdown("---")
st.markdown("## Investigation Dashboard")
st.caption("SOC triage workspace for prioritizing, assigning, documenting, and exporting local investigations.")
dashboard_reports = get_dashboard_reports()
if dashboard_reports.empty:
	st.info("No submitted reports yet. Analyze an email and submit its investigation report to populate this dashboard.")
else:
	dashboard_reports["stage"] = dashboard_reports["status_index"].apply(lambda index: TRACKING_STEPS[int(index)][0])
	dashboard_cols = st.columns(5)
	dashboard_cols[0].metric("Reports", len(dashboard_reports))
	dashboard_cols[1].metric("High risk", int((dashboard_reports["verdict"] == "FAKE / SUSPICIOUS").sum()))
	dashboard_cols[2].metric("Average risk", f"{dashboard_reports['risk_score'].mean():.0f}%")
	dashboard_cols[3].metric("Active cases", int((dashboard_reports["stage"] != "Resolved").sum()))
	dashboard_cols[4].metric("Critical", int((dashboard_reports["priority"] == "Critical").sum()))

	filter_cols = st.columns(4)
	priority_filter = filter_cols[0].selectbox("Priority", ["All", "Critical", "High", "Medium", "Low"])
	status_filter = filter_cols[1].selectbox("Case status", ["All", "Open", "Investigating", "Contained", "Closed"])
	assignee_filter = filter_cols[2].selectbox("Assignee", ["All"] + sorted(dashboard_reports["assignee"].unique().tolist()))
	search_filter = filter_cols[3].text_input("Search cases", placeholder="ID, sender, subject, or tag")
	filtered_reports = dashboard_reports.copy()
	if priority_filter != "All":
		filtered_reports = filtered_reports[filtered_reports["priority"] == priority_filter]
	if status_filter != "All":
		filtered_reports = filtered_reports[filtered_reports["case_status"] == status_filter]
	if assignee_filter != "All":
		filtered_reports = filtered_reports[filtered_reports["assignee"] == assignee_filter]
	if search_filter.strip():
		search_value = search_filter.strip().lower()
		searchable = filtered_reports[["report_id", "sender", "subject", "tags"]].fillna("").astype(str).agg(" ".join, axis=1).str.lower()
		filtered_reports = filtered_reports[searchable.str.contains(search_value, regex=False)]

	display_reports = filtered_reports.copy()
	display_reports["risk_score"] = display_reports["risk_score"].map(lambda score: f"{score}%")
	st.dataframe(
		display_reports[["report_id", "sender", "subject", "verdict", "risk_score", "priority", "case_status", "assignee", "stage", "updated_at"]],
		use_container_width=True,
		hide_index=True,
		column_config={
			"report_id": "Report ID",
			"sender": "Sender",
			"subject": "Subject",
			"verdict": "Finding",
			"risk_score": "Risk",
			"priority": "Priority",
			"case_status": "Case status",
			"assignee": "Analyst",
			"stage": "Current stage",
			"updated_at": "Last update",
		},
	)
	st.download_button(
		"⬇️  Export filtered cases as CSV",
		data=display_reports.to_csv(index=False).encode("utf-8"),
		file_name="mailsentry_case_export.csv",
		mime="text/csv",
		use_container_width=True,
	)

	st.markdown("### Analyst case controls")
	case_options = filtered_reports["report_id"].tolist() or dashboard_reports["report_id"].tolist()
	selected_case_id = st.selectbox("Select a case to triage", case_options)
	selected_case = dashboard_reports[dashboard_reports["report_id"] == selected_case_id].iloc[0]
	with st.form("case_management_form"):
		case_cols = st.columns(3)
		case_priority = case_cols[0].selectbox("Priority", ["Critical", "High", "Medium", "Low"], index=["Critical", "High", "Medium", "Low"].index(selected_case["priority"]))
		case_status = case_cols[1].selectbox("Case status", ["Open", "Investigating", "Contained", "Closed"], index=["Open", "Investigating", "Contained", "Closed"].index(selected_case["case_status"]))
		case_assignee = case_cols[2].text_input("Assigned analyst", value=selected_case["assignee"])
		case_tags = st.text_input("Tags", value=selected_case["tags"], placeholder="credential-theft, executive-impersonation")
		case_notes = st.text_area("Analyst notes", value=selected_case["analyst_notes"], placeholder="Record validation steps, affected users, or escalation context.")
		containment_action = st.text_area("Containment / response action", value=selected_case["containment_action"], placeholder="Example: blocked sender domain and reset affected credentials.")
		if st.form_submit_button("💾 Save case update", use_container_width=True):
			update_case(selected_case_id, case_priority, case_status, case_assignee, case_tags, case_notes, containment_action)
			st.success(f"Case {selected_case_id} updated.")
			st.rerun()

st.markdown("<footer>MAILSENTRY · A local educational phishing-awareness tool</footer>", unsafe_allow_html=True)
