import json
import spacy
from spacy.matcher import Matcher
import PyPDF2
from transformers import BertTokenizer, BertModel
import torch
from scipy.spatial.distance import cosine

# Initialize SpaCy and BERT
nlp = spacy.load('en_core_web_sm')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

def read_text_from_pdf(file_path):
    text = ""
    with open(file_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text()
    return text

def text_to_embedding(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    outputs = model(**inputs)
    embeddings = outputs.last_hidden_state
    mean_embedding = torch.mean(embeddings, dim=1).squeeze().detach().numpy()  
    return mean_embedding

def calculate_similarity(text1, text2, tokenizer, model):
    embedding1 = text_to_embedding(text1, tokenizer, model)
    embedding2 = text_to_embedding(text2, tokenizer, model)
    similarity = 1 - cosine(embedding1, embedding2)
    return similarity

def extract_sections(doc, matches):
    sections = {}
    end = 0
    for match_id, start, _ in matches:
        section_label = nlp.vocab.strings[match_id]
        # Find the end of the section
        section_end = next((start for _, start, _ in matches if start > end), len(doc))
        section_text = doc[end:section_end].text
        sections[section_label] = section_text.strip()
        end = section_end
    return sections

# Define patterns and matcher for SpaCy for resume
matcher_resume = Matcher(nlp.vocab)
patterns_resume = [
    {"label": "SKILLS", "pattern": [{"LOWER": "skills"}]},
    {"label": "EDUCATION", "pattern": [{"LOWER": "education"}]},
    {"label": "EXPERIENCE", "pattern": [{"LOWER": "experience"}]},
    {"label": "CERTIFICATIONS", "pattern": [{"LOWER": "certifications"}]},
    {"label": "TITLE", "pattern": [{"LOWER": "title"}]},
]
for pattern in patterns_resume:
    matcher_resume.add(pattern['label'], [pattern['pattern']])

# Read and process the resume text
resume_text = read_text_from_pdf('/home/iahsan/nodejsResume.pdf')
doc_resume = nlp(resume_text)

# Find matches and extract data for each section in resume
matches_resume = matcher_resume(doc_resume)
resume_sections = extract_sections(doc_resume, matches_resume)

# Define patterns and matcher for SpaCy for job description
matcher_job_desc = Matcher(nlp.vocab)
patterns_job_desc = [
    {"label": "SKILLS", "pattern": [{"LOWER": "skills"}]},
    {"label": "EDUCATION", "pattern": [{"LOWER": "education"}]},
    {"label": "EXPERIENCE", "pattern": [{"LOWER": "experience"}]},
    {"label": "CERTIFICATIONS", "pattern": [{"LOWER": "certifications"}]},
    {"label": "TITLE", "pattern": [{"LOWER": "title"}]},
]
for pattern in patterns_job_desc:
    matcher_job_desc.add(pattern['label'], [pattern['pattern']])

# Read and process the job description text
job_description_text = read_text_from_pdf('/home/iahsan/nodejsEdited.pdf')
doc_job_desc = nlp(job_description_text)

# Find matches and extract data for each section in job description
matches_job_desc = matcher_job_desc(doc_job_desc)
job_description_sections = extract_sections(doc_job_desc, matches_job_desc)

# print("\n======= Resume Sections =======")
# for section, text in resume_sections.items():
#     print(f"\n{section}:")
#     print(text)

# print("\n======= Job Description Sections =======")
# for section, text in job_description_sections.items():
#     print(f"\n{section}:")
#     print(text)
# Define weights for each section
weights = {
    "SKILLS": 2,
    "EDUCATION": 3,
    "EXPERIENCE": 5
}

# Initialize total weight and cumulative score
total_weight = sum(weights.values())
cumulative_score = 0

print("\n======= Similarity Scores =======")
for section, resume_text in resume_sections.items():
    if section in job_description_sections:
        job_desc_text = job_description_sections[section]
        similarity_score = calculate_similarity(resume_text, job_desc_text, tokenizer, model)
        
        # Apply weights
        weight = weights.get(section, 1)  
        scaled_similarity_score = similarity_score * (weight / total_weight)
        
#         print(f"\nSection: {section}")
#         print(f"Similarity score for {section}: {similarity_score}")
#         print(f"Weighted similarity score for {section}: {scaled_similarity_score}")
        
        cumulative_score += scaled_similarity_score

# Print the cumulative score
# print(f"\nCumulative Score: {cumulative_score}")
result = {
    "cumulative_score": cumulative_score
}

# Convert the dictionary to JSON
json_result = json.dumps(result)

# Print or return the JSON result
print(json_result)