
from llama_cpp import Llama
from transformers import pipeline
from ctransformers import AutoModelForCausalLM
import spacy
import re
from sentence_transformers import SentenceTransformer, util
import requests
from SPARQLWrapper import SPARQLWrapper, JSON
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from Levenshtein import distance as levenshtein_distance
import numpy as np
import math
from titlecase import titlecase
from spacy.lang.en.stop_words import STOP_WORDS
import multiprocessing
from multiprocessing import Pool
import sys



# Task1: Entity Recognition and Entity Linking


def ner_with_url_mapping(text):
    """
    Task: Find all entities in the answer and question text, and establish knowledge base links - Entity Query + Entity Disambiguation

    Args: - text (str): The input text containing both answer and question text.

    Returns: A dic mapping entities to their corresponding wikipeida URLs.
    """
    # Process the sentence and perform named entity recognition via Spacy
    doc = nlp(text)
    # Create a dict to store entities and their corresponding URLs
    entity_url_mapping = {}
    
    for ent in doc.ents:
        # Establish SPARQL queries to obtain candidate result sets
        entities = query_all_entities(ent.text)
        # Use distance calculation for entity disambiguation and select the entity with the minimum distance
        extracted_entity = get_most_similar_entity(ent.text, entities, text)
        print(f"The mention: {ent.text} is linked to ：{extracted_entity}")
        if extracted_entity:
          entity_url_mapping[ent.text] = extracted_entity[2]

    return entity_url_mapping

def query_all_entities(mention):
    """
    Task: Query all entities from Wikidata based on the mention str and generate a list of candidate entities.
    
    Args: - mention (str): The name of the entity.

    Returns: A list of tuples, each containing information about a possible entity:
    [(entity_label, qid, wikipedia_url, description, property_string), ...]
    """
    # Process mention to recall more candidates, performing operations like removing special characters, converting to lowercase, etc.
    mention = (
          ' '.join(mention.strip().strip("'\"").split()).lower(),
          titlecase(' '.join(mention.strip().strip("'\"").split())),
          )
    # Initialize SPARQLWrapper
    sparql = SPARQLWrapper("https://query.wikidata.org/sparql")
    # Construct SPARQL query statement, to select the entities with most common semantics, we only select 10 with most referred times 
    query = f"""
    SELECT distinct ?item ?itemLabel ?wikipediaUrl ?itemDescription ?referencedCount
    WHERE {{
      # Query 1: Direct query, e.g., apple
     {{ ?item rdfs:label "{mention[0]}"@en.
      ?article schema:about ?item;
              schema:isPartOf <https://en.wikipedia.org/>.
      BIND(IRI(CONCAT("https://en.wikipedia.org/wiki/", REPLACE(STR(?article), "https://en.wikipedia.org/wiki/", ""))) AS ?wikipediaUrl)

      ?item wikibase:sitelinks ?referencedCount.}}

      UNION
      # Query 2: Disambiguation page, e.g., apple (disambiguation)
      {{ ?temp rdfs:label "{mention[0]}"@en.
        ?temp wdt:P31 wd:Q4167410;  # Q4167410 represents a disambiguation page
           wdt:P1889 ?item.  # P642 represents the property for linking to entities
      ?article schema:about ?item;
              schema:isPartOf <https://en.wikipedia.org/>.
      BIND(IRI(CONCAT("https://en.wikipedia.org/wiki/", REPLACE(STR(?article), "https://en.wikipedia.org/wiki/", ""))) AS ?wikipediaUrl)
        
      ?item wikibase:sitelinks ?referencedCount.}}

      UNION
      # Query 3: Capitalized version, e.g., Apple
      {{ ?item rdfs:label "{mention[1]}"@en.
      ?article schema:about ?item;
              schema:isPartOf <https://en.wikipedia.org/>.
      BIND(IRI(CONCAT("https://en.wikipedia.org/wiki/", REPLACE(STR(?article), "https://en.wikipedia.org/wiki/", ""))) AS ?wikipediaUrl)

      ?item wikibase:sitelinks ?referencedCount.}}

      UNION
      # Query 4: Disambiguation page for capitalized version, e.g., Apple (disambiguation)
      {{ ?temp rdfs:label "{mention[1]}"@en.
        ?temp wdt:P31 wd:Q4167410;  # Q4167410 represents a disambiguation page
           wdt:P1889 ?item.  # P642 represents the property for linking to entities
      ?article schema:about ?item;
              schema:isPartOf <https://en.wikipedia.org/>.
      BIND(IRI(CONCAT("https://en.wikipedia.org/wiki/", REPLACE(STR(?article), "https://en.wikipedia.org/wiki/", ""))) AS ?wikipediaUrl)

      ?item wikibase:sitelinks ?referencedCount.}}

      UNION
      # Query 5: Semantic extension, e.g., China corresponding to People's Republic of China
      {{ ?temp rdfs:label "{mention[0]}"@en.
          ?temp wdt:P1889 ?item.  # P642 represents the property for linking to entities
      ?article schema:about ?item;
              schema:isPartOf <https://en.wikipedia.org/>.
      BIND(IRI(CONCAT("https://en.wikipedia.org/wiki/", REPLACE(STR(?article), "https://en.wikipedia.org/wiki/", ""))) AS ?wikipediaUrl)
    
      ?item wikibase:sitelinks ?referencedCount.}}

      UNION
        # Query 6: Semantic extension for capitalized version
      {{ ?temp rdfs:label "{mention[1]}"@en.
          ?temp wdt:P1889 ?item.  # P642 represents the property for linking to entities
      ?article schema:about ?item;
              schema:isPartOf <https://en.wikipedia.org/>.
      BIND(IRI(CONCAT("https://en.wikipedia.org/wiki/", REPLACE(STR(?article), "https://en.wikipedia.org/wiki/", ""))) AS ?wikipediaUrl)
  
      ?item wikibase:sitelinks ?referencedCount.}}

      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en" }}
    }}
    ORDER BY DESC(?referencedCount)
    LIMIT 5
    """
    # Set the SPARQL query, format, and custom HTTP header
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    sparql.addCustomHttpHeader("User-Agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3")
    try:
      # Execute the query and convert the results to JSON format
      results = sparql.query().convert()
    except Exception as e:
      print(f"querry exception：{e}")
      return []
    # Process the query results
    entities = []
    for result in results["results"]["bindings"]:
        if "disambiguation" in result.get("wikipediaUrl", {}).get("value", ""):
          continue
        entity_label = result["itemLabel"]["value"]
        qid = result["item"]["value"].split('/')[-1]
        wikipedia_url = result.get("wikipediaUrl", {}).get("value", "")
        description = result.get("itemDescription", {}).get("value", "")
        # Further query properties, for disambiguation purpose
        querry_property = f"""
        SELECT ?p ?pLabel ?propLabel ?bLabel
        WHERE
        {{
          wd:{qid} ?p ?b.

          SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
          ?prop wikibase:directClaim ?p .
        }}
        ORDER BY RAND()
        LIMIT 10

        """
        sparql.setQuery(querry_property)
        try:
          response = sparql.query().convert()
          property_string = ""
          for item in response['results']['bindings']:
              prop_label = item['propLabel']['value']
              b_label = item['bLabel']['value']

              property_string += f"[{entity_label}: {prop_label}: {b_label}]"
        except Exception as e:
          print(f"querry exception：{e}")
        entities.append((entity_label, qid, wikipedia_url, description, property_string))

    return entities

def get_most_similar_entity(mention, entities, context):
    """
    Task: Get the most similar entity based on the distance calculation between mention, list of entities, and the surrounding context.
    We also experimented with model approach but the results are not satisfactory, please see the annotation below.

    Args:
    - mention (str): The mention of the entity.
    - entities (list): A list of candidate entities, each represented as a tuple (entity_label, qid, wikipedia_url, description, property_string).
    - context (str): The surrounding context answer text/ question text.

    Returns:
    A tuple representing the most similar entity: (entity_label, qid, wikipedia_url, description, property_string)
    """
    # In case no entities found
    if not entities:
        return None
    # In case only one entity found
    if len(entities) == 1:
        return entities[0]
    # Method 1: Use a model for comparison
    # most_similar_entity = entities[0]
    # max_similarity = 0
    # model_sentence = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    # for index, entity in enumerate(entities):
    #     sentences = [entity[0]+" "+entity[3], context]
    #     embeddings = model_sentence.encode(sentences)
    #     similarity_matrix = cosine_similarity([embeddings[0]], [embeddings[1]])
    #     similarity = similarity_matrix[0][0]
    #     if similarity > max_similarity:
    #         max_similarity = similarity
    #         most_similar_entity = entities[index]
    # Method 2: Calculate distances
    distances = []
    for entity in entities:
        # Math methods are applied to adjust the parameters with longer contexts, trying to do sentence to sentence, word to word comparison here 
        distance = math.log(levenshtein_distance(context, entity[3])) * levenshtein_distance(mention, entity[0]) * math.log(levenshtein_distance(context, entity[4]))
        if distance == 0:
            return entity
        distances.append(distance)
    # Find the index of the best levenshtein distance.
    try:
        mini = np.argmin(distances)
    except (ValueError, IndexError) as e:
        print(f"An error occurred: {e}")
        mini = 0
    return entities[mini]



# Task2： Relation Extraction


def relation_extraction(text, question_text, entity_url_mapping):
    """
    Task: Extract relation triplets from the given text and return the triplet (best answer) with the highest semantic similarity to the question text.

    Args:
    - text (str): The answer text from which relation triplets will be extracted.
    - question_text (str): The question text used for semantic similarity comparison.
    - entity_url_mapping (dict): A dict mapping entity mentions to their corresponding URLs.

    Returns:
    A tuple representing the relation triplet with the highest semantic similarity: (subject, relation, object).
    """
    # Use a text-to-text generation model for relation extraction, the model has been pre-trained with Wikidata dataset, thus the relations extracted are comparable
    model = pipeline("text2text-generation", model="ibm/knowgl-large")
    doc = nlp(text)
    triplets = []
    # Split the answer text into sentences
    sentences = [sent.text for sent in doc.sents]
    # Add the (complete answer text + question text) as a single entity for extraction
    sentences.append(text + question_text)
    # Process each sentence, extract relation triplets and collect with the right format
    for sentence in sentences:
        model_result = model(sentence)[0]['generated_text']
        results = re.findall(r'\[.*?\]', model_result)
        for result in results:
            parts = result[1:-1].split('|')
            parts[0] = parts[0].split('#')[0].replace('(', '')  
            parts[2] = parts[2].split('#')[0].replace('(', '')  
            triplets.append(tuple(parts))
    # Compare triplets and select the one most semantically similar to the question text

    # Method 1: Model judgment
    # Commented out to avoid potential dependency issues
    # most_similar_triplet = triplets[0]
    # max_similarity = 0
    # model_sentence = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    # for index, triplet in enumerate(triplets):
    #     triplet_text = ' '.join(triplet)
    #     sentences = [triplet_text, question_text]
    #     embeddings = model_sentence.encode(sentences)
    #     similarity_matrix = cosine_similarity([embeddings[0]], [embeddings[1]])
    #     similarity = similarity_matrix[0][0]
    #     if similarity > max_similarity:
    #         max_similarity = similarity
    #         most_similar_triplet = triplets[index]

    #  Method 2: Calculate distances

    # Process the triplets list to filter noises
    # In case we filter all the candidates, need to set a backup to avoid errors, the last added whole-context triplet is selected
    try:
        backup = triplets[-1]
    except (ValueError, IndexError) as e:
        print(f"An error occurred: {e}")
        backup = ("","","")
    # Filter1: Remove the triplets with the relation of "instance of", as they are mostly over-interpreted, seldomly appears in the real sentences
    triplets = [triplet for triplet in triplets if triplet[1] != "instance of"]
    # Filter2: Both entities in the triplets must be in the mapped entity_url_mapping
    triplets = [triplet for triplet in triplets if (triplet[0] in entity_url_mapping.keys() and triplet[2] in entity_url_mapping.keys()) ]
    # Again, distance calculation for similarity comparsion
    distances = []
    for triplet in triplets:
      triplet_text = ' '.join(triplet)
      distance = math.log(levenshtein_distance(triplet_text, question_text)) * levenshtein_distance(triplet[0], question_text) * levenshtein_distance(triplet[1], question_text) 
      if distance == 0:
        return triplet
      distances.append(distance)
    try:
        mini = np.argmin(distances)
        most_similar_triplet = triplets[mini]
    except (ValueError, IndexError) as e:
        print(f"An error occurred: {e}")
        most_similar_triplet = backup
        
    return most_similar_triplet


# Task3： Fact Checking in a Knowledge Base


def url_qid(wikipedia_url):
    """
    Task: Convert a Wikipedia URL to the corresponding Wikidata entity QID.

    Args:
    - wikipedia_url (str): The URL of the Wikipedia page.

    Returns:
    A string representing the Wikidata QID or None if not found.
    """
    # Extract the title of the Wikipedia page
    title = wikipedia_url.split("/")[-1]
    # Build the Wikidata API request
    api_url = "https://www.wikidata.org/w/api.php"
    params = {
        "action": "wbgetentities",
        "sites": "enwiki",  
        "titles": title,
        "format": "json"
    }
    # Send the request
    response = requests.get(api_url, params=params)
    data = response.json()
    # Extract QID from the response
    entities = data.get("entities", {})
    if entities:
        entity = next(iter(entities.values()))
        qid = entity.get("id")
        return qid
    
    return None


def to_plabel(property_id):
    """
    Task: Convert a Wikidata property ID to the corresponding property label.

    Args:
    - property_id (str): The Wikidata property ID.

    Returns:
    A string representing the English label of the property or None if not found.
    """
    if "/" in property_id:
      property_id = property_id.split("/")[-1]
    # Build Wikidata request for property info
    api_url = "https://www.wikidata.org/w/api.php"
    params = {
        "action": "wbgetentities",
        "props": "labels",
        "ids": property_id,
        "format": "json"
    }
    # Send the request
    response = requests.get(api_url, params=params)
    data = response.json()
    # Extract property name from the response
    entities = data.get("entities", {})
    if entities:
        entity = entities.get(property_id, {})
        labels = entity.get("labels", {})
        english_label = labels.get("en", {}).get("value")
        return english_label
    
    return None


def wiki_verify(triplet, entity_url_mapping):
    """
    Task: Verify a triplet by querying Wikidata to check if the given predicate is a valid relation between entities.

    Args:
    - triplet (tuple): A tuple representing the triplet (entity1, predicate, entity2).
    - entity_url_mapping (dict): A dict mapping entity names to their corresponding Wikipedia URLs.

    Returns:
    True if the predicate is a valid relation between entities, False otherwise.
    """
    entity1, predicate, entity2 = triplet
    url_1 = entity_url_mapping.get(entity1)
    url_2 = entity_url_mapping.get(entity2)
    # Check if URLs exist
    if not url_1 or not url_2:
        return False
    # Get QID based on URLs
    qid_1 = url_qid(url_1)
    qid_2 = url_qid(url_2)

    sparql = SPARQLWrapper("https://query.wikidata.org/sparql")
    # Build SPARQL query
    sparql.setQuery(f"""
        SELECT ?relation
        WHERE
        {{
            {{
              # Query direct relation from e1 to e2
              wd:{qid_1} ?relation wd:{qid_2} .
              }}
            UNION
            {{
              # Query direct relation from e2 to e1
              wd:{qid_2} ?relation wd:{qid_1} .
              }}
            UNION
            {{
              # Query parent relation of the e1 to e2 relation
              wd:{qid_1} ?childrelation wd:{qid_2} .
              ?childrelation wdt:P1647 ?relation .
            }}
            UNION
            {{
              # Query parent relation of the e2 to e1 relation
              wd:{qid_2} ?childrelation wd:{qid_1} .
              ?childrelation wdt:P1647 ?relation .
            }}
            SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en" }} .
        }}
        """)
    sparql.setReturnFormat(JSON)
    sparql.addCustomHttpHeader("User-Agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3")
    results = sparql.query().convert()

    # Check if the query was successful
    if 'results' in results and 'bindings' in results['results']:
        # Process query results, compare after lemmatization
        doc_p = nlp(predicate)
        for token in doc_p:
          if token.lemma_ not in STOP_WORDS:
            predicate = token.lemma_
        relation_labels = list(map(to_plabel, [result['relation']['value'] for result in results['results']['bindings']]))
        doc_r = nlp(" ".join(relation_labels))
        # Perform lemmatization and filter stop words
        relation_labels = [token.lemma_ for token in doc_r if token.lemma_ not in STOP_WORDS]
        # match the result list with the original predicate
        return predicate in relation_labels
    else:
        print("Query failed.")
        return False


# Other tasks: Sentence Parsing


def is_special_question(sentence):
    """
    Task: Check if a given sentence is a special question(need to be answered with an entity).

    Args:
    - sentence (str): The input sentence to be checked.

    Returns:
    True if the sentence is a special interrogative sentence, False otherwise.
    """
    doc = nlp(sentence)
    # Check if it does not start with an auxiliary verb
    not_start_with_aux = not (len(doc) > 0 and doc[0].pos_ == "AUX")
    # alternative but does not include "xxx is ...": doc[0].pos_ == "PRON" and doc[0].text.lower() in ["who", "what", "where", "which", "when", "why", "how"]
    return not_start_with_aux


def contains_negation_spacy(sentences, question):
    """
    Task: Check if the answer to a general question is negative based on spaCy pattern rules.
    Because the output of LLM is unsatisfactory with many irrelevant and conflict sentences, 
    we conclude this by anlyse the sentence in the text that has the most similar semantics to the question. 

    Args:
    - sentences (str): The input text generated by LLM.
    - question (str): The original question.

    Returns:
    True if the answer is considered negative, False otherwise.
    """
    # Seperate the whole text into sentences
    doc = nlp(sentences)
    sentences = [sent.text for sent in doc.sents]
    # Find the sentence most similar to the question text
    distances = []
    for sentence in sentences:
        distance = math.log(levenshtein_distance(sentence, question)) 
        distances.append(distance)
    try:
        mini = np.argmin(distances)
    except (ValueError, IndexError) as e:
        print(f"An error occurred: {e}")
        mini = 0
    answer_sentence = sentences[mini]
    doc = nlp(answer_sentence)
    # Check if the sentence contains a token with dependency label "neg"
    for token in doc:
        if token.dep_ == "neg":
            return True
    
    return False


# Parallel the Program as a Complete Task


def run_task(question):
    """
    Task: Execute the entire parallelized task.

    Args:
    - question (tuple): A tuple containing question_id and question_text.

    Returns:
    A tuple with information about the task execution:
    - number (int): The question number for later sorting.
    - question_id (str): The ID of the question.
    - answer_LLM (str): The answer generated by LLM.
    - extracted_answer (str): The extracted answer based on the question type.
    - correctness (str): The correctness status of the extracted answer.
    - entity_url_mapping (dict): A dict mapping entities to their URLs.
    """
    question_id, question_text = question
    # Invoke LLM
    print(f"LLM is generating the answer for {question_id} ... Please wait")
    answer_LLM = llm(question_text)
    # Perform entity recognition and entity linking
    print(f"Performing Entity Recognition and Lingking for {question_id}")
    entity_url_mapping = ner_with_url_mapping(question_text + answer_LLM)
    # Handle different types of questions:
    # For special questions
    print("Further processing and Generating results, please wait ... ")
    if is_special_question(question_text):
        # Extract the most matching relation triplet from the answer
        extracted_spo = relation_extraction(answer_LLM, question_text, entity_url_mapping)
        # The output answer is the second entity of the triplet
        extracted_answer = entity_url_mapping.get(extracted_spo[2], None)
        # Validate correctness using knowledge base
        if wiki_verify(extracted_spo, entity_url_mapping):
            correctness = "correct"
        else:
            correctness = "incorrect"
    # For general questions
    else:
        # Check for negation of the answer and output YES/NO
        assertion = contains_negation_spacy(answer_LLM, question_text)
        if assertion:
            extracted_answer = "No"
        else:
            extracted_answer = "Yes"
        # Extract the most matching relation triplet from the question
        extracted_spo = relation_extraction(question_text, question_text, entity_url_mapping)
        # Validate correctness using knowledge base: parent-child relationships are also considered
        if wiki_verify(extracted_spo, entity_url_mapping) == assertion:
            correctness = "incorrect"
        else:
            correctness = "correct"
    # Process the question number for sorting purpose
    number = int(question_id.lstrip('question-'))

    return number, question_id, answer_LLM, extracted_answer, correctness, entity_url_mapping


# Run the System

# Set IO Path, read the path from command-line
if len(sys.argv) != 3:
    print("Usage: python wdps.py input_path output_path.")
    confirm = input("Do you want to use the default path setting, otherwise leave?: y/n  ")
    if confirm.lower() in {"y", "yes"}:
        input_file_path = './input.txt'  # Replace with the actual path to your file
        output_file_path = './output.txt'
    else:
        raise RuntimeError("System stopped!") 
else:
    input_file_path = sys.argv[1]  
    output_file_path = sys.argv[2]

# load a model for Spacy
nlp = spacy.load("en_core_web_sm")
# Set the Configuration of LLM
repository="TheBloke/Llama-2-7B-GGUF"
model_file="llama-2-7b.Q4_K_M.gguf"
llm = AutoModelForCausalLM.from_pretrained(repository, model_file=model_file, model_type="llama")


# Read the question from the text file by lines
print("Loading the question file ...")
input = {}
with open(input_file_path, 'r', encoding='utf-8') as file:
    for line in file:
        # Split the line based on the tab character
        parts = line.strip().split('\t')
        # Ensure the line is properly formatted
        if len(parts) == 2:
            question_id, question_text = parts
            if question_text.startswith("Question:"):
                question_text = question_text[len("Question: "):].strip()
            if question_text.endswith("Answer:"):
                question_text = question_text[:-len("Answer: ")].strip()
            input[question_id] = question_text
        else:
            print(f"Invalid line format: {line}")
print("Successful! Distributing computation tasks ...", input)
# Set the process number
cpu_num = multiprocessing.cpu_count()
print(f"{cpu_num} cpus in computation")
pool = Pool(processes=cpu_num) 
# Parallelize the program, each question will be handled as a task sent to a processor
results = pool.map(run_task, [(question_id, question_text) for question_id, question_text in input.items()])  
# We also experimented the dask framework for parallelization
# results = dask.compute(*delayed_tasks, scheduler='processes') 
# Sort the results with the preprocessed question number
print("Successful! Processing the results ...")
results = sorted(results, key=lambda x: x[0])
# Result processing
for result in results:
    _, question_id, answer_LLM, extracted_answer, correctness, entity_url_mapping = result
    with open(output_file_path, 'a', encoding='utf-8') as file:
            file.write(f'{question_id}\tR"{answer_LLM}"\n')
            file.write(f'{question_id}\tA"{extracted_answer}"\n')
            file.write(f'{question_id}\tC"{correctness}"\n')
            for key, value in entity_url_mapping.items():
                file.write(f'{question_id}\tE"{key}"\t"{value}"\n')
    # Also print results on the Console
    print("-----------------------------------------------------------")
    print(f'{question_id}\tR"{answer_LLM}"\n')
    print(f'{question_id}\tA"{extracted_answer}"\n')
    print(f'{question_id}\tC"{correctness}"')
    for key, value in entity_url_mapping.items():
        print(f'{question_id}\tE"{key}"\t"{value}"')
print("Successful! Please check out the output.txt.")
pool.close()