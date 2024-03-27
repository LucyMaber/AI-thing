# from nltk import word_tokenize, pos_tag
# import nltk
# from nltk.corpus import wordnet as wn
# from nltk.corpus import wordnet
# from nltk.stem import WordNetLemmatizer
# from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize
def init_nlp():
    nltk.download('wordnet')
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('stopwords')
# init_nlp()


def pos_tag_text(text):
    """
    Perform part-of-speech tagging on the given text.

    Args:
        text (str): The input text to be tagged.

    Returns:
        list of tuples: A list of (word, pos_tag) tuples.
    """
    words = word_tokenize(text)
    pos_tags = pos_tag(words)
    return pos_tags


def remove_stop_words(text):
    # Tokenize the text
    words = word_tokenize(text)

    # Get the list of English stop words
    stop_words = set(stopwords.words('english'))

    # Remove stop words
    filtered_words = [word for word in words if word.lower() not in stop_words]

    # Recreate the text without stop words
    text_without_stop_words = ' '.join(filtered_words)

    return text_without_stop_words


def nominalize(text):
    output = ""
    lemmatized_words = []

    lemmatizer = WordNetLemmatizer()

    def get_wordnet_pos(tag):
        if tag.startswith('J'):
            return wordnet.ADJ
        elif tag.startswith('V'):
            return wordnet.VERB
        elif tag.startswith('N'):
            return wordnet.NOUN
        elif tag.startswith('R'):
            return wordnet.ADV
        else:
            return wordnet.NOUN  # Default to noun
    for i in pos_tag_text(text):
        wordnet_pos = get_wordnet_pos(i[1][0])
        lemma = lemmatizer.lemmatize(i[0], wordnet_pos)
        lemmatized_words.append(lemma)
    return remove_stop_words(" ".join(lemmatized_words)).lower()


