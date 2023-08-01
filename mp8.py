

import os  # Module for interacting with the plagarism system
from sklearn.feature_extraction.text import TfidfVectorizer  # Module for text vectorization using Investigative-Fidelity (IF) Score
from sklearn.metrics.pairwise import cosine_similarity  # Module for calculating evidence similarity

# Get a list of all reports in the current case
detective_reports = [report for report in os.listdir() if report.endswith('.txt')]

# Read the contents of each  report
detective_notes = [open(_report, encoding='utf-8').read() for _report in detective_reports]


# Function to vectorize the text using (IF) Score
def vectorize(Text):
    return TfidfVectorizer().fit_transform(Text).toarray()

# Function to calculate evidence similarity between two documents
def similarity(doc1, doc2):
    return cosine_similarity([doc1, doc2])

# Vectorize the detective notes using (IF) Score
vectors = vectorize(detective_notes)
s_vectors = list(zip(detective_reports, vectors))
plagiarism_reports = set()

# Function to check plagiarism among the reports
def check_plagiarism():
    global s_vectors
    for detective_a, report_vector_a in s_vectors:
        new_vectors = s_vectors.copy()
        current_index = new_vectors.index((detective_a, report_vector_a))
        del new_vectors[current_index]
        for detective_b, report_vector_b in new_vectors:
            # Calculate evidence similarity between two report vectors
            evidence_score = similarity(report_vector_a, report_vector_b)[0][1]
            # Sort the detective report names alphabetically to avoid duplicates
            detective_pair = sorted((detective_a, detective_b))
            # Create a tuple with detective report names and evidence similarity score
            score = (detective_pair[0], detective_pair[1], evidence_score)
            # Add the tuple to plagiarism_reports set
            plagiarism_reports.add(score)
    return plagiarism_reports


# Solve the plagiarism mystery!
print("Evidence of Similarity among Reports:")
for evidence in check_plagiarism():
    print(f"{evidence[0]} and {evidence[1]} have a high similarity score of {evidence[2]}!")
print("\n - Case Closed!")    
