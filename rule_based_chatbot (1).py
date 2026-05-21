"""
╔══════════════════════════════════════════════════════════════╗
║              Rule-Based AI Chatbot  🤖                       ║
║                                                              ║
║  Architecture  : Modular intent-matching engine              ║
║  NLP Pipeline  : Tokenization → Normalization → Matching     ║
║  Features      : Sentiment detection, memory, math engine,   ║
║                  conversation logging, colorized UI           ║
╚══════════════════════════════════════════════════════════════╝
"""

import re
import os
import sys
import json
import math
import random
import datetime
import textwrap
from collections import defaultdict

try:
    from colorama import init, Fore, Style
    init(autoreset=True)
    COLOR = True
except ImportError:
    COLOR = False
    class Fore:
        GREEN = CYAN = YELLOW = RED = MAGENTA = BLUE = WHITE = ""
    class Style:
        BRIGHT = RESET_ALL = DIM = ""

try:
    import nltk
    for _pkg in ["punkt_tab", "stopwords"]:
        try:
            nltk.data.find(f"tokenizers/{_pkg}" if "punkt" in _pkg else f"corpora/{_pkg}")
        except LookupError:
            nltk.download(_pkg, quiet=True)
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    STOP_WORDS = set(stopwords.words("english"))
    NLP = True
except Exception:
    NLP = False
    STOP_WORDS = set()


# ══════════════════════════════════════════════════════════════
#  SECTION 1 — NLP Preprocessor
# ══════════════════════════════════════════════════════════════

class Preprocessor:
    """Cleans and normalises raw user input before intent matching."""

    CONTRACTIONS = {
        "i'm": "i am", "i've": "i have", "i'll": "i will", "i'd": "i would",
        "you're": "you are", "you've": "you have", "you'll": "you will",
        "he's": "he is", "she's": "she is", "it's": "it is",
        "we're": "we are", "they're": "they are", "what's": "what is",
        "that's": "that is", "there's": "there is", "who's": "who is",
        "can't": "cannot", "won't": "will not", "don't": "do not",
        "doesn't": "does not", "didn't": "did not", "isn't": "is not",
        "aren't": "are not", "wasn't": "was not", "weren't": "were not",
    }

    @staticmethod
    def expand_contractions(text):
        for c, e in Preprocessor.CONTRACTIONS.items():
            text = text.replace(c, e)
        return text

    @staticmethod
    def normalize(text):
        text = text.lower().strip()
        text = Preprocessor.expand_contractions(text)
        text = re.sub(r"[^\w\s\+\-\*\/\.\=\?]", "", text)
        text = re.sub(r"\s+", " ", text)
        return text

    @staticmethod
    def tokenize(text):
        if NLP:
            tokens = word_tokenize(text)
            return [t for t in tokens if t not in STOP_WORDS and len(t) > 1]
        return text.split()

    @staticmethod
    def process(raw):
        normalized = Preprocessor.normalize(raw)
        tokens = Preprocessor.tokenize(normalized)
        return {"raw": raw, "normalized": normalized, "tokens": tokens}


# ══════════════════════════════════════════════════════════════
#  SECTION 2 — Sentiment Analyzer
# ══════════════════════════════════════════════════════════════

class SentimentAnalyzer:
    """Lightweight lexicon-based sentiment detection."""

    POSITIVE = {
        "good","great","awesome","fantastic","wonderful","happy","love","like",
        "nice","excellent","amazing","thanks","thank","perfect","beautiful",
        "brilliant","fine","okay","ok","cool"
    }
    NEGATIVE = {
        "bad","awful","terrible","hate","horrible","sad","angry","upset",
        "unhappy","worst","poor","wrong","stupid","broken","annoying",
        "useless","disappointed","frustrated","boring"
    }

    @staticmethod
    def analyze(tokens):
        token_set = set(tokens)
        pos = len(token_set & SentimentAnalyzer.POSITIVE)
        neg = len(token_set & SentimentAnalyzer.NEGATIVE)
        if pos > neg:
            return "positive"
        elif neg > pos:
            return "negative"
        return "neutral"


# ══════════════════════════════════════════════════════════════
#  SECTION 3 — Math Engine
# ══════════════════════════════════════════════════════════════

class MathEngine:
    """Parses and evaluates mathematical expressions safely."""

    NATURAL = [
        (r"add\s+([\d.]+)\s+and\s+([\d.]+)",          lambda m: float(m[0]) + float(m[1])),
        (r"subtract\s+([\d.]+)\s+from\s+([\d.]+)",    lambda m: float(m[1]) - float(m[0])),
        (r"multiply\s+([\d.]+)\s+by\s+([\d.]+)",      lambda m: float(m[0]) * float(m[1])),
        (r"divide\s+([\d.]+)\s+by\s+([\d.]+)",        lambda m: float(m[0]) / float(m[1]) if float(m[1]) != 0 else None),
        (r"([\d.]+)\s*(?:percent|%)\s*of\s*([\d.]+)", lambda m: (float(m[0]) / 100) * float(m[1])),
        (r"square\s+root\s+of\s+([\d.]+)",            lambda m: math.sqrt(float(m[0]))),
        (r"([\d.]+)\s+squared",                       lambda m: float(m[0]) ** 2),
        (r"([\d.]+)\s+cubed",                         lambda m: float(m[0]) ** 3),
    ]

    SAFE = {k: getattr(math, k) for k in dir(math) if not k.startswith("_")}
    SAFE.update({"abs": abs, "round": round, "min": min, "max": max})

    @staticmethod
    def evaluate(text):
        for pattern, func in MathEngine.NATURAL:
            m = re.search(pattern, text)
            if m:
                result = func(m.groups())
                if result is None:
                    return None, "Division by zero is undefined."
                return result, None
        expr = re.sub(r"[^0-9\+\-\*\/\(\)\.\%\*\s]", "", text).strip()
        if expr and re.search(r"\d", expr):
            try:
                result = eval(expr, {"__builtins__": {}}, MathEngine.SAFE)
                return result, None
            except Exception:
                pass
        return None, "Could not parse that expression."


# ══════════════════════════════════════════════════════════════
#  SECTION 4 — Conversation Memory
# ══════════════════════════════════════════════════════════════

class Memory:
    """Stores session context, history, intent counts and mood."""

    def __init__(self):
        self.user_name = None
        self.history = []
        self.intent_counts = defaultdict(int)
        self.mood_history = []
        self.session_start = datetime.datetime.now()
        self.turn_count = 0

    def remember(self, role, text, intent=None, sentiment=None):
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        self.history.append({"role": role, "text": text, "time": ts})
        if intent:
            self.intent_counts[intent] += 1
        if sentiment:
            self.mood_history.append(sentiment)
        if role == "user":
            self.turn_count += 1

    def dominant_mood(self):
        if not self.mood_history:
            return "neutral"
        return max(set(self.mood_history), key=self.mood_history.count)

    def session_duration(self):
        delta = datetime.datetime.now() - self.session_start
        mins, secs = divmod(int(delta.total_seconds()), 60)
        return f"{mins}m {secs}s"

    def export_log(self, path="chat_log.json"):
        data = {
            "session_start": self.session_start.isoformat(),
            "session_duration": self.session_duration(),
            "user_name": self.user_name or "Anonymous",
            "turns": self.turn_count,
            "dominant_mood": self.dominant_mood(),
            "intent_summary": dict(self.intent_counts),
            "history": self.history,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        return path


# ══════════════════════════════════════════════════════════════
#  SECTION 5 — Intent Rule Base
# ══════════════════════════════════════════════════════════════

INTENT_RULES = [
    {
        "intent": "greeting",
        "patterns": [
            r"\b(hello|hi|hey|howdy|greetings|sup|yo)\b",
            r"\bgood (morning|afternoon|evening|day)\b",
        ],
        "responses": {
            "default": [
                "Hello{name}! Great to meet you. I'm RuleBot. Type 'help' to see what I can do.",
                "Hey{name}! I'm ready to chat. What's on your mind?",
                "Hi{name}! How can I assist you today?",
            ],
            "returning": [
                "Welcome back{name}! What can I do for you?",
                "Hey again{name}! Still here and ready to help.",
            ],
        },
    },
    {
        "intent": "wellbeing_ask",
        "patterns": [
            r"\bhow are you\b", r"\bhow do you do\b",
            r"\bare you okay\b", r"\bhow is it going\b",
        ],
        "responses": {
            "default": [
                "Running at full capacity — no bugs today! How about you?",
                "Doing great, thanks for asking! How are you?",
                "Excellent as always! 😄 What can I do for you?",
            ]
        },
    },
    {
        "intent": "user_wellbeing_good",
        "patterns": [r"\bi am (fine|good|great|okay|ok|doing well|fantastic)\b"],
        "responses": {"default": ["Glad to hear it! What can I do for you? 😊"]},
    },
    {
        "intent": "user_wellbeing_bad",
        "patterns": [r"\bi am (sad|unhappy|upset|not (good|okay|well|great)|terrible|bad)\b"],
        "responses": {
            "default": [
                "I'm sorry to hear that 😢. I hope things improve soon!",
                "That sounds tough. Every storm runs out of rain 🌈. I'm here!",
            ]
        },
    },
    {
        "intent": "bot_name",
        "patterns": [
            r"\bwhat is your name\b", r"\bwho are you\b",
            r"\bwhat should i call you\b",
        ],
        "responses": {
            "default": [
                "I'm RuleBot 🤖 — a rule-based AI chatbot built in Python!",
                "Call me RuleBot! Powered by intent-matching rules and Python magic. ✨",
            ]
        },
    },
    {
        "intent": "bot_age",
        "patterns": [r"\bhow old are you\b", r"\bwhen were you (born|created|built|made)\b"],
        "responses": {"default": ["Fresh out of the Python interpreter! 🐣 Very new."]},
    },
    {
        "intent": "bot_creator",
        "patterns": [r"\bwho (made|created|built|programmed|coded) you\b"],
        "responses": {
            "default": [
                "A talented developer built me from scratch using Python. Their rule-based masterpiece!",
            ]
        },
    },
    {
        "intent": "help",
        "patterns": [r"\b(help|commands|options|what can you do|capabilities|features)\b"],
        "responses": {"default": ["__HELP__"]},
    },
    {
        "intent": "math",
        "patterns": [
            r"\b(calculate|compute|solve|what is|whats)\b.*[\d\+\-\*\/]",
            r"\badd\s+[\d.]+\s+and\s+[\d.]+",
            r"\bsubtract\s+[\d.]+\s+from\s+[\d.]+",
            r"\bmultiply\s+[\d.]+\s+by\s+[\d.]+",
            r"\bdivide\s+[\d.]+\s+by\s+[\d.]+",
            r"\b[\d.]+\s*[\+\-\*\/]\s*[\d.]+",
            r"\bsquare root of\b",
            r"\b[\d.]+\s+(squared|cubed|percent)\b",
        ],
        "handler": "math",
    },
    {
        "intent": "time",
        "patterns": [r"\b(what is the time|current time|what time is it|time now)\b"],
        "handler": "time",
    },
    {
        "intent": "date",
        "patterns": [r"\b(what is (the |today)date|what day is (it|today)|today date|todays date)\b"],
        "handler": "date",
    },
    {
        "intent": "joke",
        "patterns": [r"\b(tell me a joke|joke|make me laugh|something funny)\b"],
        "handler": "joke",
    },
    {
        "intent": "quote",
        "patterns": [r"\b(quote|inspire me|motivate|motivation|inspiration|wise words)\b"],
        "handler": "quote",
    },
    {
        "intent": "fact",
        "patterns": [r"\b(fun fact|tell me a fact|random fact|interesting fact|did you know)\b"],
        "handler": "fact",
    },
    {
        "intent": "flip_coin",
        "patterns": [r"\b(flip (a )?coin|heads or tails|coin flip|toss (a )?coin)\b"],
        "handler": "coin",
    },
    {
        "intent": "roll_dice",
        "patterns": [r"\b(roll (a )?dice|roll (a )?die|dice roll)\b"],
        "handler": "dice",
    },
    {
        "intent": "stats",
        "patterns": [r"\b(stats|statistics|session info|how long have we|how many (messages|turns))\b"],
        "handler": "stats",
    },
    {
        "intent": "save_log",
        "patterns": [r"\b(save (log|chat|history)|export (log|chat)|log (this|chat|session))\b"],
        "handler": "save_log",
    },
    {
        "intent": "thanks",
        "patterns": [r"\b(thank(s| you)|thx|cheers|appreciate it|ty)\b"],
        "responses": {
            "default": [
                "You're welcome{name}! 😊 Anything else I can help with?",
                "Happy to help! That's what I'm here for. 🤖",
                "Anytime! Ask me anything.",
            ]
        },
    },
    {
        "intent": "farewell",
        "patterns": [
            r"\b(bye|goodbye|see you|later|farewell|take care|good night|cya|quit|exit)\b",
        ],
        "responses": {
            "default": [
                "Goodbye{name}! It was great chatting with you. Take care! 👋",
                "See you later{name}! Stay curious. 🌟",
                "Bye{name}! Come back anytime — I'll be right here. 🤖",
            ]
        },
        "exit": True,
    },
]

JOKES = [
    "Why don't scientists trust atoms?\nBecause they make up everything! 😄",
    "Why did the programmer quit his job?\nBecause he didn't get arrays! 😂",
    "How do you comfort a JavaScript bug?\nYou console it. 😏",
    "Why do Python programmers prefer dark mode?\nBecause light attracts bugs! 🐛",
    "A SQL query walks into a bar...\n'Can I JOIN you?' 🍺",
    "What did the ocean say to the beach?\nNothing, it just waved. 🌊",
]

QUOTES = [
    '"The only way to do great work is to love what you do." — Steve Jobs',
    '"In the middle of every difficulty lies opportunity." — Albert Einstein',
    '"Code is like humor. When you have to explain it, it\'s bad." — Cory House',
    '"First, solve the problem. Then, write the code." — John Johnson',
    '"Any fool can write code a computer understands. Good programmers write code humans understand." — Martin Fowler',
]

FACTS = [
    "🧠 The human brain has about 86 billion neurons.",
    "🐙 Octopuses have three hearts and blue blood.",
    "💻 The first computer bug was an actual moth — found in the Harvard Mark II in 1947.",
    "🔢 Python was named after Monty Python's Flying Circus, not the snake.",
    "🌐 The first website is still online at info.cern.ch.",
    "🐝 Honey never spoils — edible honey was found in 3000-year-old Egyptian tombs.",
]

HELP_TEXT = """
┌─────────────────────────────────────────────────────────┐
│                 RuleBot — Command Guide                  │
├───────────────┬─────────────────────────────────────────┤
│  Category     │  Example inputs                         │
├───────────────┼─────────────────────────────────────────┤
│  Greetings    │  hi, hello, hey, good morning           │
│  Identity     │  my name is Sara                        │
│  About bot    │  who are you, how old are you           │
│  Feelings     │  how are you, i am sad                  │
│  Math         │  add 5 and 3  |  10 * 4 + 2            │
│               │  square root of 144  |  20% of 500      │
│  Date / Time  │  what time is it  |  today's date       │
│  Fun          │  tell me a joke  |  inspire me          │
│               │  fun fact  |  flip a coin  |  roll dice  │
│  Session      │  stats  |  save log                     │
│  Exit         │  bye  |  quit  |  exit  |  goodbye      │
└───────────────┴─────────────────────────────────────────┘"""


# ══════════════════════════════════════════════════════════════
#  SECTION 6 — Response Engine
# ══════════════════════════════════════════════════════════════

class ResponseEngine:
    """
    Core engine: matches processed input against the rule base
    and returns context-aware, sentiment-personalised responses.
    """

    def __init__(self, memory):
        self.memory = memory

    def match_intent(self, normalized):
        for rule in INTENT_RULES:
            for pattern in rule["patterns"]:
                m = re.search(pattern, normalized)
                if m:
                    return rule, m
        return None, None

    def format_name(self, template):
        if self.memory.user_name:
            return template.replace("{name}", f" {self.memory.user_name}")
        return template.replace("{name}", "")

    def pick_response(self, rule, sentiment):
        responses = rule.get("responses", {})
        pool = responses.get(sentiment) or responses.get("default") or []
        return random.choice(pool) if pool else None

    def handle(self, raw_input):
        """Returns (response_text, intent_name, should_exit)"""
        processed = Preprocessor.process(raw_input)
        normalized = processed["normalized"]
        tokens = processed["tokens"]
        sentiment = SentimentAnalyzer.analyze(tokens)

        # ── Name capture ──────────────────────────────────────
        for np in [r"\bmy name is (\w+)\b", r"\bcall me (\w+)\b", r"\bthey call me (\w+)\b"]:
            m = re.search(np, normalized)
            if m:
                self.memory.user_name = m.group(1).capitalize()
                return (
                    f"Nice to meet you, {self.memory.user_name}! 😊 "
                    "I'll remember that. What can I do for you?",
                    "set_name", False
                )

        rule, match = self.match_intent(normalized)

        if rule is None:
            return (
                random.choice([
                    "I'm not sure I understand. Try 'help' to see what I can do.",
                    "Hmm, that's outside my rules. Could you rephrase?",
                    "I didn't quite get that. I work best with clear inputs — try 'help'!",
                ]),
                "unknown", False
            )

        intent = rule["intent"]
        should_exit = rule.get("exit", False)
        handler = rule.get("handler")

        # ── Handlers ──────────────────────────────────────────
        if handler == "math":
            result, error = MathEngine.evaluate(normalized)
            if error:
                return f"⚠️  {error}", intent, False
            fmt = int(result) if isinstance(result, float) and result.is_integer() else round(result, 6)
            return f"🔢 Result: {fmt}", intent, False

        if handler == "time":
            now = datetime.datetime.now().strftime("%I:%M:%S %p")
            return f"⏰ Current time: {now}", intent, False

        if handler == "date":
            today = datetime.datetime.now().strftime("%A, %B %d, %Y")
            return f"📅 Today is: {today}", intent, False

        if handler == "joke":
            return random.choice(JOKES), intent, False

        if handler == "quote":
            return random.choice(QUOTES), intent, False

        if handler == "fact":
            return random.choice(FACTS), intent, False

        if handler == "coin":
            return f"🪙 It's... {random.choice(['Heads', 'Tails'])}!", intent, False

        if handler == "dice":
            return f"🎲 You rolled a {random.randint(1, 6)}!", intent, False

        if handler == "stats":
            m = self.memory
            top = max(m.intent_counts, key=m.intent_counts.get) if m.intent_counts else "N/A"
            return (
                f"📊 Session Stats\n"
                f"   Duration    : {m.session_duration()}\n"
                f"   Turns       : {m.turn_count}\n"
                f"   Your mood   : {m.dominant_mood()}\n"
                f"   Top intent  : {top}\n"
                f"   Name on file: {m.user_name or 'not set'}"
            ), intent, False

        if handler == "save_log":
            path = self.memory.export_log()
            return f"💾 Conversation saved to '{path}'", intent, False

        if intent == "help":
            return HELP_TEXT, intent, False

        # ── Standard responses ────────────────────────────────
        response = self.pick_response(rule, sentiment)
        if response:
            response = self.format_name(response)
            if intent == "greeting" and self.memory.turn_count > 1:
                pool = rule["responses"].get("returning") or rule["responses"].get("default")
                response = self.format_name(random.choice(pool))
            return response, intent, should_exit

        return "Understood, but I have no response for that yet.", intent, should_exit


# ══════════════════════════════════════════════════════════════
#  SECTION 7 — Terminal UI
# ══════════════════════════════════════════════════════════════

class TerminalUI:
    W = 62

    @staticmethod
    def clear():
        os.system("cls" if os.name == "nt" else "clear")

    @staticmethod
    def banner():
        b = [
            "╔" + "═" * (TerminalUI.W - 2) + "╗",
            "║" + "  Rule-Based AI Chatbot 🤖".center(TerminalUI.W - 2) + "║",
            "║" + "  NLP · Intent Matching · Memory · Math".center(TerminalUI.W - 2) + "║",
            "╚" + "═" * (TerminalUI.W - 2) + "╝",
        ]
        print(Fore.CYAN + Style.BRIGHT + "\n".join(b))
        print(Fore.WHITE + Style.DIM + "  Type 'help' for commands | 'bye' to exit\n")

    @staticmethod
    def bot_say(text, intent=""):
        prefix = Fore.GREEN + Style.BRIGHT + "  🤖 Bot  : " + Style.RESET_ALL
        lines = text.split("\n")
        for i, line in enumerate(lines):
            wrapped = textwrap.fill(line, width=TerminalUI.W + 12, subsequent_indent="           ")
            if i == 0:
                print(prefix + wrapped)
            else:
                print(Fore.WHITE + "           " + wrapped)
        if intent and intent not in ("unknown", "help"):
            print(Fore.WHITE + Style.DIM + f"           [intent: {intent}]")
        print()

    @staticmethod
    def user_prompt(name=""):
        label = f"  👤 {name or 'You'}   : "
        try:
            return input(Fore.YELLOW + Style.BRIGHT + label + Style.RESET_ALL)
        except (EOFError, KeyboardInterrupt):
            return "exit"

    @staticmethod
    def divider():
        print(Fore.WHITE + Style.DIM + "  " + "─" * (TerminalUI.W - 2))

    @staticmethod
    def farewell(memory):
        print()
        print(Fore.CYAN + "  ╔" + "═" * (TerminalUI.W - 4) + "╗")
        print(Fore.CYAN + "  ║" + " Session Summary ".center(TerminalUI.W - 4) + "║")
        print(Fore.CYAN + "  ╚" + "═" * (TerminalUI.W - 4) + "╝")
        print(Fore.WHITE + f"  Duration : {memory.session_duration()}")
        print(Fore.WHITE + f"  Turns    : {memory.turn_count}")
        print(Fore.WHITE + f"  Mood     : {memory.dominant_mood()}")
        print()


# ══════════════════════════════════════════════════════════════
#  SECTION 8 — Main Loop
# ══════════════════════════════════════════════════════════════

def main():
    ui = TerminalUI()
    memory = Memory()
    engine = ResponseEngine(memory)

    ui.clear()
    ui.banner()

    ui.bot_say(
        "Hello! I'm RuleBot 🤖\n"
        "I use NLP preprocessing, sentiment detection, a math engine,\n"
        "and conversation memory to chat with you.\n"
        "What's your name?",
        "greeting"
    )

    while True:
        try:
            raw = ui.user_prompt(memory.user_name).strip()
        except KeyboardInterrupt:
            print()
            ui.bot_say("Session interrupted. Goodbye! 👋")
            break

        if not raw:
            ui.bot_say("Please type something — I'm listening! 👂")
            continue

        memory.remember("user", raw)
        response, intent, should_exit = engine.handle(raw)
        memory.remember(
            "bot", response, intent=intent,
            sentiment=SentimentAnalyzer.analyze(Preprocessor.tokenize(raw))
        )

        ui.divider()
        ui.bot_say(response, intent)

        if should_exit:
            ui.farewell(memory)
            break

    print(Fore.CYAN + Style.DIM + "  [RuleBot session ended]\n")


if __name__ == "__main__":
    main()
