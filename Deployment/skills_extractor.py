import spacy
import en_core_web_sm
from collections import Counter

# Load spaCy model once
try:
    nlp = en_core_web_sm.load()
except:
    import spacy.cli
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

# Comprehensive dictionary of skills categorized by domain
CATEGORY_SKILLS = {
    'Data Science': ['python', 'r', 'sql', 'machine learning', 'deep learning', 'pandas', 'numpy', 'scikit-learn', 'tensorflow', 'pytorch', 'statistics', 'nlp', 'data visualization', 'tableau', 'powerbi'],
    'HR': ['recruitment', 'onboarding', 'employee relations', 'talent acquisition', 'performance management', 'hris', 'payroll', 'conflict resolution', 'interviewing', 'sourcing', 'workday', 'peoplesoft'],
    'Advocate': ['litigation', 'legal research', 'drafting', 'contract law', 'corporate law', 'negotiation', 'court proceedings', 'legal advising', 'dispute resolution', 'compliance'],
    'Arts': ['graphic design', 'illustration', 'adobe creative suite', 'photoshop', 'illustrator', 'typography', 'visual arts', 'painting', 'sculpting', 'art history'],
    'Web Designing': ['html', 'css', 'javascript', 'ui/ux', 'figma', 'adobe xd', 'responsive design', 'bootstrap', 'wireframing', 'prototyping', 'web typography'],
    'Mechanical Engineer': ['autocad', 'solidworks', 'thermodynamics', 'fluid mechanics', 'manufacturing', 'cad/cam', 'fea', 'ansys', 'robotics', 'hvac', 'matlab'],
    'Sales': ['b2b', 'b2c', 'cold calling', 'crm', 'salesforce', 'lead generation', 'negotiation', 'closing', 'account management', 'presentation skills', 'sales strategy'],
    'Health and fitness': ['personal training', 'nutrition', 'anatomy', 'physiology', 'first aid', 'cpr', 'fitness assessment', 'kinesiology', 'rehabilitation', 'diet planning'],
    'Civil Engineer': ['autocad', 'structural analysis', 'surveying', 'construction management', 'geotechnical engineering', 'staad.pro', 'revit', 'project management', 'concrete technology', 'estimation'],
    'Java Developer': ['java', 'spring boot', 'hibernate', 'j2ee', 'restful apis', 'microservices', 'maven', 'junit', 'sql', 'git', 'tomcat', 'oop'],
    'Business Analyst': ['requirements gathering', 'sql', 'excel', 'agile', 'scrum', 'data analysis', 'stakeholder management', 'jira', 'process modeling', 'business intelligence', 'tableau'],
    'SAP Developer': ['sap abap', 'sap hana', 'sap fiori', 'sap erp', 'odata', 'sap bw', 'sap mm', 'sap sd', 'sap fico', 'bapi', 'smartforms'],
    'Automation Testing': ['selenium', 'java', 'python', 'testng', 'cucumber', 'appium', 'api testing', 'postman', 'jenkins', 'ci/cd', 'maven', 'jira', 'git'],
    'Electrical Engineering': ['matlab', 'circuit design', 'autocad electrical', 'power systems', 'plc', 'scada', 'simulink', 'control systems', 'microcontrollers', 'electronics'],
    'Operations Manager': ['process improvement', 'supply chain', 'logistics', 'six sigma', 'lean management', 'project management', 'inventory management', 'budgeting', 'team leadership', 'kpi tracking'],
    'Python Developer': ['python', 'django', 'flask', 'fastapi', 'rest api', 'sql', 'postgresql', 'docker', 'kubernetes', 'aws', 'git', 'linux', 'celery'],
    'DevOps Engineer': ['aws', 'azure', 'docker', 'kubernetes', 'jenkins', 'ci/cd', 'terraform', 'ansible', 'linux', 'bash', 'git', 'prometheus', 'grafana'],
    'Network Security Engineer': ['firewall', 'cisco', 'wireshark', 'vpn', 'penetration testing', 'ceh', 'ids/ips', 'cryptography', 'network routing', 'linux', 'tcp/ip'],
    'PMO': ['project management', 'agile', 'scrum', 'pmp', 'jira', 'confluence', 'risk management', 'budgeting', 'stakeholder management', 'ms project', 'reporting'],
    'Database': ['sql', 'mysql', 'postgresql', 'oracle', 'mongodb', 'database design', 'pl/sql', 'nosql', 'etl', 'data modeling', 'query optimization'],
    'Hadoop': ['hadoop', 'spark', 'hive', 'pig', 'scala', 'java', 'hdfs', 'mapreduce', 'sqoop', 'kafka', 'big data', 'hbase'],
    'ETL Developer': ['etl', 'informatica', 'sql', 'data warehousing', 'ssis', 'talend', 'data pipeline', 'python', 'oracle', 'data modeling', 'aws redshift'],
    'DotNet Developer': ['c#', '.net', 'asp.net', 'mvc', 'entity framework', 'sql server', 'web api', 'linq', 'wcf', 'azure', 'visual studio', 'javascript'],
    'Blockchain': ['solidity', 'smart contracts', 'ethereum', 'web3.js', 'cryptography', 'rust', 'hyperledger', 'dapps', 'bitcoin', 'go', 'consensus algorithms'],
    'Testing': ['manual testing', 'qa', 'test case design', 'jira', 'bug tracking', 'regression testing', 'agile', 'sql', 'api testing', 'postman', 'sdlc'],
    'Frontend': ['html', 'css', 'javascript', 'react', 'vue.js', 'angular', 'typescript', 'redux', 'tailwind', 'bootstrap', 'webpack', 'ui/ux'],
    'Backend': ['python', 'java', 'node.js', 'express', 'django', 'spring boot', 'sql', 'nosql', 'rest api', 'graphql', 'docker', 'aws'],
    'Full Stack': ['react', 'node.js', 'express', 'mongodb', 'html', 'css', 'javascript', 'python', 'django', 'sql', 'git', 'docker'],
}

def flatten_skills():
    """Create a flat list of all known skills for general extraction."""
    all_skills = set()
    for skills in CATEGORY_SKILLS.values():
        for skill in skills:
            all_skills.add(skill.lower())
    return list(all_skills)

ALL_KNOWN_SKILLS = flatten_skills()

def extract_skills_from_text(text, predicted_category=None):
    """
    Extracts skills using spaCy NER and keyword matching.
    Returns: top skills found, missing skills for the category, and a match score.
    """
    text_lower = text.lower()
    doc = nlp(text_lower)
    
    found_skills = set()
    
    # 1. Keyword matching against known skills
    for skill in ALL_KNOWN_SKILLS:
        if skill in text_lower:
            # Basic boundary check to avoid partial word matches (e.g. 'c' in 'cat')
            import re
            pattern = r'\b' + re.escape(skill) + r'\b'
            if re.search(pattern, text_lower):
                found_skills.add(skill)
                
    # 2. Extract specific entities that might be skills (e.g., ORG, PRODUCT)
    # Often tech stacks get classified as PRODUCT or ORG by spaCy
    for ent in doc.ents:
        if ent.label_ in ['PRODUCT', 'ORG']:
            # Filter out common false positives by checking length or keeping it raw
            clean_ent = ent.text.strip()
            if len(clean_ent) > 2 and len(clean_ent) < 20:
                found_skills.add(clean_ent)
                
    found_skills_list = list(found_skills)
    
    # Calculate match score and missing skills if category is known
    match_score = 0
    missing_skills = []
    category_expected_skills = []
    
    if predicted_category:
        # Find the closest category key (case-insensitive)
        matched_cat_key = None
        for key in CATEGORY_SKILLS.keys():
            if key.lower() == predicted_category.lower() or predicted_category.lower() in key.lower():
                matched_cat_key = key
                break
                
        if matched_cat_key:
            category_expected_skills = CATEGORY_SKILLS[matched_cat_key]
            matched_count = 0
            for req_skill in category_expected_skills:
                if req_skill.lower() in found_skills:
                    matched_count += 1
                else:
                    missing_skills.append(req_skill)
            
            if len(category_expected_skills) > 0:
                match_score = int((matched_count / len(category_expected_skills)) * 100)
        else:
            # If category is completely unknown, we default to a generic score
            match_score = min(100, len(found_skills) * 5)
            
    # Format skills for output
    top_10_skills = [skill.title() for skill in found_skills_list[:10]]
    missing_skills_formatted = [skill.title() for skill in missing_skills[:10]]
    
    return {
        "top_skills": top_10_skills,
        "missing_skills": missing_skills_formatted,
        "match_score": match_score
    }
