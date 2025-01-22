import unittest
import random
import re
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from typing import List, Optional
import csv

nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('punkt', quiet=True)
STOPWORDS = set(stopwords.words('english'))

class AdversarialGenerator:
    def __init__(
        self,
        passages_file: str,
        injection_sentences_file: str = None,
        instruction: str = "",
        queries_file: str = None
    ):
        """
        Initialize the AdversarialGenerator with necessary components.

        Parameters:
        - passages_file (str): Path to a TSV file containing passages with 'id' and 'passage' columns.
        - injection_sentences_file (str, optional): Path to a text file containing injection sentences, one per line.
        - instruction (str): Optional instruction to prepend to queries.
        - queries_file (str, optional): Path to a text file containing queries, one per line.
        """
        self.instruction = instruction
        self.passages = []
        self.injection_sentences = []
        self.queries = []

        # Load passages from the provided TSV file
        with open(passages_file, 'r', encoding='utf-8') as f:
            tsv_reader = csv.DictReader(f, delimiter='\t')
            for row in tsv_reader:
                passage_id = row['id']
                passage_text = row['passage'].strip()
                self.passages.append(passage_text)

        # Load injection sentences from the provided file
        if injection_sentences_file:
            with open(injection_sentences_file, 'r', encoding='utf-8') as f:
                self.injection_sentences = [line.strip() for line in f]

        # Load queries from the provided file
        if queries_file:
            with open(queries_file, 'r', encoding='utf-8') as f:
                self.queries = [line.strip() for line in f if line.strip()]

    def sample_random_passage(self) -> str:
        """
        Sample a random passage from the loaded passages.

        Returns:
        - str: the passage text.
        """
        if not self.passages:
            raise ValueError("No passages available to sample from.")
        return random.choice(self.passages)
        
    def extract_keywords(self, text: str) -> List[str]:
        """
        Extract keywords from text by removing stopwords and non-alphabetic characters.

        Parameters:
        - text (str): Text from which to extract keywords.

        Returns:
        - List[str]: A list of keywords.
        """
        words = word_tokenize(text.lower())
        keywords = [word for word in words if word not in STOPWORDS]
        return keywords

    def expand_keywords_with_synonyms(self, keywords: List[str], num_synonyms_per_word: int = 2) -> List[str]:
        """
        Expand the list of keywords by adding synonyms from WordNet.

        Parameters:
        - keywords (List[str]): Original list of keywords.
        - num_synonyms_per_word (int): Number of synonyms to add for each word.

        Returns:
        - List[str]: Expanded list of keywords with synonyms included.
        """
        expanded_keywords = set(keywords)
        for word in keywords:
            synsets = wordnet.synsets(word)
            synonyms = set()
            for synset in synsets:
                for lemma in synset.lemmas():
                    synonym = lemma.name().replace('_', ' ')
                    if synonym != word and synonym not in expanded_keywords:
                        synonyms.add(synonym)
                    if len(synonyms) >= num_synonyms_per_word:
                        break
                if len(synonyms) >= num_synonyms_per_word:
                    break
            expanded_keywords.update(synonyms)
        return list(expanded_keywords)

    def get_random_sentences(self, num_sentences: int = 1) -> List[str]:
        """
        Retrieve a random set of sentences from the injection sentences.

        Parameters:
        - num_sentences (int): Number of sentences to retrieve.

        Returns:
        - List[str]: A list of randomly selected sentences.
        """
        return random.sample(self.injection_sentences, k=num_sentences)

    def form_text_from_random_words(self, num_words: int = 100) -> str:
        """
        Form a text by taking random words from randomly selected sentences.

        Parameters:
        - num_words (int): Number of words to include in the text.

        Returns:
        - str: The formed text.
        """
        text_words = []
        while len(text_words) < num_words:
            sentence = random.choice(self.injection_sentences)
            words = word_tokenize(sentence)
            if words:
                word = random.choice(words)
                text_words.append(word)
        return ' '.join(text_words[:num_words])
    
    def prepend_query_to_passage(self, query: str, passage: str) -> str:
        """
        Prepend a query to the passage to generate an adversarial passage.

        Parameters:
        - query (str): The query.
        - passage (str): The original passage.

        Returns:
        - str: Modified passage with 'Query: {query}' prepended.
        """
        if not query:
            raise ValueError("No query given. Please provide a query.")
        return f"Query: {query}\n{passage}"

    def inject_sentences(
        self, text: str, inject_sentences: List[str], location: Optional[str] = None
    ) -> str:
        """
        Inject sentences into a text at a specified location.

        Parameters:
        - text (str): Original text.
        - inject_sentences (List[str]): Sentences to inject.
        - location (str, optional): Where to inject sentences ('start', 'middle', 'end', or None).

        Returns:
        - str: Modified text
        """
        valid_locations = {'start', 'middle', 'end', None}
        if location not in valid_locations:
            raise ValueError("Invalid location. Choose from 'start', 'middle', 'end', or None.")

        if location is None:
            location = random.choice(['start', 'middle', 'end'])
        
        def modify_sentence(sentence):
            """Randomly remove period or uncapitalize the first word."""
            if random.random() < 0.5:  # 50% chance to remove period
                sentence = sentence.rstrip('.')
            
            if random.random() < 0.5:  # 50% chance to uncapitalize the first word
                # Split the sentence into words
                words = sentence.split()
                if words:  # Ensure there is at least one word
                    # Check if the first word is capitalized (first letter is uppercase, rest are lowercase)
                    first_word = words[0]
                    if first_word[0].isupper() and first_word[1:].islower():
                        # Uncapitalize the first word
                        words[0] = first_word[0].lower() + first_word[1:]
                        sentence = ' '.join(words)
            return sentence

        inject_sentences = [modify_sentence(sentence) for sentence in inject_sentences]

        if location == 'start':
            modified_text = ' '.join(inject_sentences) + ' ' + text
        elif location == 'end':
            modified_text = text + ' ' + ' '.join(inject_sentences)
        elif location == 'middle':
            words = text.split()
            for sentence in inject_sentences:
                if len(words) > 1:
                    insert_index = random.randint(1, len(words) - 1)
                else:
                    insert_index = 0
                words.insert(insert_index, sentence)
            modified_text = ' '.join(words)

        return modified_text

    def remove_sentence(self, text: str) -> str:
        """
        Remove the first or last sentence from the text.

        Parameters:
        - text (str): The original text.

        Returns:
        - str: Text with one sentence removed.
        """
        sentences = sent_tokenize(text)
        if len(sentences) <= 1:
            # Can't remove a sentence if there's only one
            return text
        if random.randint(0, 1):
            del sentences[0]
        else:
            del sentences[-1]
        return ' '.join(sentences)

    def inject_query_keywords_into_passage(
        self, passage: str, query: str, location: Optional[str] = 'middle', expand: bool = False, num_synonyms_per_word: int = 2, shorten_passage: bool = False,
    ) -> str:
        """
        Inject query keywords into a passage.

        Parameters:
        - passage (str): The original passage.
        - query (str): The query text.
        - location (str, optional): Where to inject keywords ('start', 'middle', 'end', or None).
        - expand (bool): Whether to expand keywords with synonyms.
        - shorten_passage (bool): Whether to shorten the passage by removing a sentence.

        Returns:
        - str: Modified text
        """
        if shorten_passage:
            passage = self.remove_sentence(passage)
        query_keywords = self.extract_keywords(query)
        if expand:
            query_keywords = self.expand_keywords_with_synonyms(query_keywords, num_synonyms_per_word=num_synonyms_per_word)
        return self.inject_sentences(passage, query_keywords, location)

    def inject_query_into_passage(
        self, passage: str, query: str, location: Optional[str] = 'middle', num_injections: int = 1, shorten_passage: bool = False,
    ) -> str:
        """
        Inject a query into a passage.

        Parameters:
        - passage (str): The original passage.
        - query (str): The query to inject.
        - location (str, optional): Where to inject the query ('start', 'middle', 'end', or None).
        - shorten_passage (bool): Whether to shorten the passage by removing a sentence.

        Returns:
        - str: Modified text
        """
        if shorten_passage:
            passage = self.remove_sentence(passage)
        injected_passage = passage
        for _ in range(num_injections):
            injected_passage = self.inject_sentences(injected_passage, [query], location)

        return injected_passage

    def inject_random_sentences_into_text(
        self, text: str, num_sentences: int = 1, location: Optional[str] = None, shorten_passage: bool = False,
    ) -> str:
        """
        Inject random sentences into a text.

        Parameters:
        - text (str): The original text.
        - num_sentences (int): Number of sentences to inject.
        - location (str, optional): Where to inject sentences ('start', 'middle', 'end', or None).
        - shorten_passage (bool): Whether to shorten the passage by removing a sentence.

        Returns:
        - str: Modified text
        """
        if shorten_passage:
            text = self.remove_sentence(text)

        inject_sentences = self.get_random_sentences(num_sentences)
        return self.inject_sentences(text, inject_sentences, location)

    def inject_random_sentences_into_passage(
        self, passage: str, num_sentences: int = 1, location: Optional[str] = None, shorten_passage: bool = False,
    ) -> str:
        """
        Inject random sentences into a passage.

        Parameters:
        - passage (str): The original passage.
        - num_sentences (int): Number of sentences to inject.
        - location (str, optional): Where to inject sentences ('start', 'middle', 'end', or None).

        Returns:
        - str: Modified text
        """
        return self.inject_random_sentences_into_text(passage, num_sentences, location, shorten_passage)

    def inject_random_sentences_into_query(
        self, query: str, num_sentences: int = 1, location: Optional[str] = None, include_instruction: bool = True
    ) -> str:
        """
        Inject random sentences into a query.

        Parameters:
        - query (str): The query text.
        - num_sentences (int): Number of sentences to inject.
        - location (str, optional): Where to inject sentences ('start', 'middle', 'end', or None).
        - include_instruction (bool): Whether to include the instruction.

        Returns:
        - str: Modified text
        """
        if include_instruction:
            query = self.instruction + query
        return self.inject_random_sentences_into_text(query, num_sentences, location)

class TestAdversarialGenerator(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        # Initialize with sample data
        self.generator = AdversarialGenerator(
            passages_file='passages/msmarco.tsv',
            injection_sentences_file='random_sentences/msmarco.txt',
            instruction='',
            queries_file='queries/queries.txt'
        )
        self.passage = "The quick brown fox jumps over the lazy dog."
        self.query = "What is the color of the fox?"

    def print_separator(self):
        print("\n" + "-" * 50 + "\n")

    def test_expand_keywords_with_synonyms(self):
        self.print_separator()
        print("\nRunning test:", self._testMethodName)
        keywords = self.generator.extract_keywords(self.query)
        expanded_keywords = self.generator.expand_keywords_with_synonyms(keywords)
        print("Original Keywords:", keywords)
        print("Expanded Keywords with Synonyms:", expanded_keywords)
        self.assertGreater(len(expanded_keywords), len(keywords))

    def test_inject_query_into_passage(self):
        self.print_separator()
        print("\nRunning test:", self._testMethodName)
        modified_passage = self.generator.inject_query_into_passage(self.passage, self.query)
        print("Original Passage:", self.passage)
        print("Query:", self.query)
        print("Modified Passage (with Query Injected):", modified_passage)
        self.assertIn(self.query, modified_passage)

    def test_inject_random_sentences_into_passage(self):
        self.print_separator()
        print("\nRunning test:", self._testMethodName)
        modified_passage = self.generator.inject_random_sentences_into_passage(self.passage, num_sentences=2)
        print("Original Passage:", self.passage)
        print("Modified Passage (with Random Sentences Injected):", modified_passage)
        self.assertNotEqual(modified_passage, self.passage)

    def test_extract_keywords(self):
        self.print_separator()
        print("\nRunning test:", self._testMethodName)
        text = "The quick brown fox jumps over the lazy dog."
        keywords = self.generator.extract_keywords(text)
        print("Text:", text)
        print("Extracted Keywords:", keywords)
        expected_keywords = ['quick', 'brown', 'fox', 'jumps', 'lazy', 'dog']
        self.assertEqual(set(keywords), set(expected_keywords))

    def test_form_text_from_random_words(self):
        self.print_separator()
        print("\nRunning test:", self._testMethodName)
        num_words = 50
        text = self.generator.form_text_from_random_words(num_words)
        word_count = len(text.split())
        print("Generated Text:", text)
        print("Number of Words:", word_count)
        self.assertEqual(word_count, num_words)

    def test_inject_query_keywords_into_passage(self):
        self.print_separator()
        print("\nRunning test:", self._testMethodName)
        passage = self.passage
        query = self.query
        modified_passage = self.generator.inject_query_keywords_into_passage(passage, query)
        print("Original Passage:", passage)
        print("Query:", query)
        print("Modified Passage:", modified_passage)
        query_keywords = self.generator.extract_keywords(query)
        for keyword in query_keywords:
            self.assertIn(keyword, modified_passage)

    def test_inject_random_sentences_into_text(self):
        self.print_separator()
        print("\nRunning test:", self._testMethodName)
        text = "This is a sample text for testing."
        num_sentences = 2
        modified_text = self.generator.inject_random_sentences_into_text(text, num_sentences)
        print("Original Text:", text)
        print("Modified Text:", modified_text)
        self.assertNotEqual(modified_text, text)

    def test_inject_random_sentences_into_query(self):
        self.print_separator()
        print("\nRunning test:", self._testMethodName)
        query = self.query
        num_sentences = 2
        modified_query = self.generator.inject_random_sentences_into_query(query, num_sentences)
        print("Original Query:", query)
        print("Modified Query:", modified_query)
        self.assertNotEqual(modified_query, query)

if __name__ == '__main__':
    unittest.main()