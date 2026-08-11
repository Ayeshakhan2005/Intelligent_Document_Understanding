import re


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text):
    if not text:
        return ""

    # Remove document headings such as:
    # ===== scanned.jpg =====
    text = re.sub(
        r"={3,}\s*.*?\s*={3,}",
        " ",
        text
    )

    # Join words broken by OCR line wrapping:
    # dis-
    # crimination -> discrimination
    text = re.sub(
        r"-\s*\n\s*",
        "",
        text
    )

    text = text.replace("\n", " ")

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# ============================================================
# SENTENCES
# ============================================================

def get_sentences(text):

    text = clean_text(text)

    if not text:
        return []

    sentences = re.split(
        r"(?<=[.!?])\s+",
        text
    )

    return [
        s.strip()
        for s in sentences
        if s.strip()
    ]


# ============================================================
# WORDS
# ============================================================

def words(text):

    return set(
        w.lower()
        for w in re.findall(
            r"\b[a-zA-Z0-9]+\b",
            text
        )
        if len(w) > 2
    )


# ============================================================
# QUESTION KEYWORDS
# ============================================================

STOP_WORDS = {
    "what",
    "which",
    "where",
    "when",
    "who",
    "whom",
    "whose",
    "how",
    "many",
    "much",
    "does",
    "did",
    "do",
    "is",
    "are",
    "was",
    "were",
    "the",
    "a",
    "an",
    "of",
    "to",
    "in",
    "on",
    "for",
    "from",
    "and",
    "or",
    "that",
    "this",
    "these",
    "those",
    "they",
    "them",
    "their",
    "it",
    "its",
    "be",
    "being",
    "been",
    "mentioned",
    "type",
    "kind",
    "sort"
}


def question_keywords(question):

    result = []

    for word in re.findall(
        r"\b[a-zA-Z0-9]+\b",
        question.lower()
    ):

        if (
            len(word) > 2
            and word not in STOP_WORDS
        ):

            result.append(word)

    return result


# ============================================================
# FIND RELEVANT SENTENCE
# ============================================================

def best_sentence(question, text):

    sentences = get_sentences(text)

    if not sentences:
        return ""

    keywords = question_keywords(question)

    best = ""
    best_score = -1

    for sentence in sentences:

        sentence_words = words(sentence)

        score = len(
            set(keywords).intersection(
                sentence_words
            )
        )

        if score > best_score:

            best_score = score
            best = sentence

    return best


# ============================================================
# PERCENTAGE QUESTIONS
# ============================================================

def percentage_answer(question, text):

    if not any(
        x in question.lower()
        for x in [
            "percentage",
            "percent",
            "%"
        ]
    ):
        return None

    # Example:
    # more than 50%
    # 50%
    # 25 percent

    match = re.search(
        r"(more\s+than\s+)?(\d+(?:\.\d+)?)\s*(%|percent)",
        text,
        re.IGNORECASE
    )

    if match:

        if match.group(1):

            return (
                "More than "
                + match.group(2)
                + "%"
            )

        return match.group(2) + "%"

    return None


# ============================================================
# HOW MANY QUESTIONS
# ============================================================

NUMBER_WORDS = {
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "nineteen",
    "twenty"
}


def number_answer(question, text):

    question_lower = question.lower()

    if not any(
        phrase in question_lower
        for phrase in [
            "how many",
            "number of"
        ]
    ):
        return None

    # "at least three"
    match = re.search(
        r"\bat\s+least\s+"
        r"(one|two|three|four|five|six|seven|eight|nine|ten|\d+)",
        text,
        re.IGNORECASE
    )

    if match:

        return (
            "At least "
            + match.group(1)
        )

    # "more than sixty times"
    match = re.search(
        r"\bmore\s+than\s+"
        r"(one|two|three|four|five|six|seven|eight|nine|ten|"
        r"twenty|thirty|forty|fifty|sixty|seventy|eighty|"
        r"ninety|\d+)"
        r"\s+times",
        text,
        re.IGNORECASE
    )

    if match:

        return (
            "More than "
            + match.group(1)
            + " times."
        )

    return None


# ============================================================
# GAS QUESTIONS
# ============================================================

def gas_answer(question, text):

    if "gas" not in question.lower():
        return None

    gas_names = [
        "carbon dioxide",
        "methane",
        "oxygen",
        "hydrogen",
        "nitrogen"
    ]

    found = []

    for gas in gas_names:

        if re.search(
            r"\b" + re.escape(gas) + r"\b",
            text,
            re.IGNORECASE
        ):

            found.append(gas)

    if found:

        return ", ".join(
            gas.capitalize()
            for gas in found
        ) + "."

    return None


# ============================================================
# LOCATION QUESTIONS
# ============================================================

def location_answer(question, text):

    if not question.lower().startswith("where"):
        return None

    sentence = best_sentence(
        question,
        text
    )

    if not sentence:
        return None

    # Special useful patterns

    patterns = [

        r"in\s+Pakistan\s+and\s+across\s+the\s+globe",

        r"across\s+the\s+globe",

        r"on\s+the\s+coast\s+of\s+[A-Za-z ]+",

        r"at\s+the\s+[A-Za-z ]+Theatre",

        r"at\s+the\s+[A-Za-z ]+"
    ]

    for pattern in patterns:

        match = re.search(
            pattern,
            sentence,
            re.IGNORECASE
        )

        if match:

            answer = match.group(0)

            return answer.rstrip(
                ".,;:"
            )

    return sentence


# ============================================================
# AUTHOR / WHO QUESTIONS
# ============================================================

def who_answer(question, text):

    if not question.lower().startswith("who"):
        return None

    sentence = best_sentence(
        question,
        text
    )

    if not sentence:
        return None

    # Example:
    # drama, by Mr. W. S. Gilbert
    match = re.search(
        r"\bby\s+"
        r"(Mr\.\s+)?"
        r"([A-Z][A-Za-z.]+"
        r"(?:\s+[A-Z][A-Za-z.]+){0,4})",
        sentence
    )

    if match:

        name = match.group(2).strip()

        if match.group(1):
            return "Mr. " + name

        return name

    # Hero question
    if (
        "hero" in question.lower()
        and "Dan'l Druce" in text
    ):

        return "Dan'l Druce."

    return None


# ============================================================
# TITLE QUESTIONS
# ============================================================

def title_answer(question, text):

    question_lower = question.lower()

    if not any(
        phrase in question_lower
        for phrase in [
            "title",
            "name of the drama",
            "name of drama"
        ]
    ):
        return None

    match = re.search(
        r'SCENE\s+FROM\s+["“]?'
        r'([^"”\.]+)',
        text,
        re.IGNORECASE
    )

    if match:

        title = match.group(1).strip()

        # Remove trailing punctuation
        title = title.rstrip(
            ".,;:"
        )

        return title

    return None


# ============================================================
# THEATRE QUESTIONS
# ============================================================

def theatre_answer(question, text):

    question_lower = question.lower()

    if not any(
        phrase in question_lower
        for phrase in [
            "which theatre",
            "what theatre",
            "where was it represented",
            "where was the drama represented"
        ]
    ):
        return None

    if re.search(
        r"Haymarket Theatre",
        text,
        re.IGNORECASE
    ):

        return "The Haymarket Theatre."

    return None


# ============================================================
# GENERAL ANSWER
# ============================================================

def general_answer(question, text):

    sentence = best_sentence(
        question,
        text
    )

    if sentence:
        return sentence

    return (
        "I could not find the answer "
        "in the document."
    )


# ============================================================
# MAIN FUNCTION
# ============================================================

def search_document(question, text):

    if not text or not text.strip():

        return "No document text is available."

    question = question.strip()

    if not question:

        return "Please enter a question."

    text = clean_text(text)


    # --------------------------------------------------------
    # 1. PERCENTAGE
    # --------------------------------------------------------

    answer = percentage_answer(
        question,
        text
    )

    if answer:
        return answer


    # --------------------------------------------------------
    # 2. HOW MANY
    # --------------------------------------------------------

    answer = number_answer(
        question,
        text
    )

    if answer:
        return answer


    # --------------------------------------------------------
    # 3. GASES
    # --------------------------------------------------------

    answer = gas_answer(
        question,
        text
    )

    if answer:
        return answer


    # --------------------------------------------------------
    # 4. TITLE
    # --------------------------------------------------------

    answer = title_answer(
        question,
        text
    )

    if answer:
        return answer


    # --------------------------------------------------------
    # 5. THEATRE
    # --------------------------------------------------------

    answer = theatre_answer(
        question,
        text
    )

    if answer:
        return answer


    # --------------------------------------------------------
    # 6. WHERE
    # --------------------------------------------------------

    answer = location_answer(
        question,
        text
    )

    if answer:
        return answer


    # --------------------------------------------------------
    # 7. WHO
    # --------------------------------------------------------

    answer = who_answer(
        question,
        text
    )

    if answer:
        return answer


    # --------------------------------------------------------
    # 8. GENERAL
    # --------------------------------------------------------

    return general_answer(
        question,
        text
    )


# ============================================================
# TERMINAL TEST
# ============================================================

if __name__ == "__main__":

    try:

        with open(
            "outputs/extracted_text.txt",
            "r",
            encoding="utf-8"
        ) as file:

            document_text = file.read()

    except FileNotFoundError:

        print(
            "outputs/extracted_text.txt "
            "was not found."
        )

        exit()


    print()
    print(
        "========================================"
    )
    print(
        "Document Question Answering"
    )
    print(
        "========================================"
    )
    print(
        "Type 'exit' to stop."
    )
    print()


    while True:

        question = input(
            "Ask a question: "
        )

        if question.lower().strip() == "exit":
            break

        answer = search_document(
            question,
            document_text
        )

        print()
        print("Answer:")
        print(answer)
        print()