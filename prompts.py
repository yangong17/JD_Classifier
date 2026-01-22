"""
Prompt templates and scoring rubrics for job description classification.
"""

# List of compensable factors - must match EXACTLY what the LLM returns
FACTORS = [
    "Knowledge and Skills",
    "Problem Solving and Complexity",
    "Decision Authority",
    "Impact and Organizational Scope",
    "Stakeholder Interaction and Influence",
    "Experience",
    "Supervisory Responsibility",
    "Budget and Resource Accountability",
    "Working Conditions",
]

# Mapping from factor names to desired CSV column names
FACTOR_COLUMN_NAMES = {
    "Knowledge and Skills": "Knowledge_Skills",
    "Problem Solving and Complexity": "Problem_Solving",
    "Decision Authority": "Decision_Authority",
    "Impact and Organizational Scope": "Impact_Org_Scope",
    "Stakeholder Interaction and Influence": "Stakeholder_Influence",
    "Experience": "Experience",
    "Supervisory Responsibility": "Supervisory_Responsibility",
    "Budget and Resource Accountability": "Budget_Resource_Accountability",
    "Working Conditions": "Working_Conditions",
}

# Phase 1: Binary check prompt
PHASE1_PROMPT = """Analyze the following job description and determine if it contains information related to these 9 compensable factors.
Respond with "Yes" if there is explicit or strongly implied information, and "No" if the information is missing, vague, or cannot be inferred.

Factors:
1. Knowledge & Skills - Depth and breadth of specific expertise, professional and technical knowledge required.
2. Problem Solving & Complexity - Level of analytical thinking, judgement, and complex decision-making required.
3. Decision Authority - Degree of autonomy and discretion to make decisions that affect outcomes.
4. Impact & Organizational Scope - Breadth of impact — whether a role affects a small team, department, or the whole enterprise.
5. Stakeholder Interaction & Influence - Degree and complexity of engagement with internal and external partners.
6. Experience - Amount and relevance of prior experience necessary to perform the job effectively.
7. Supervisory Responsibility - Leadership and managerial scope — number of direct reports and responsibility for team performance.
8. Budget & Resource Accountability - Responsibility for financial resources, asset oversight, or allocation decisions.
9. Working Conditions - Environmental or contextual demands associated with the job.

{pre_prompt}

Job Description:
{job_description}

{post_prompt}

Instructions:
- Respond ONLY in the exact format below, one factor per line.
- Use "Yes" or "No" only.

Knowledge & Skills: [Yes/No]
Problem Solving & Complexity: [Yes/No]
Decision Authority: [Yes/No]
Impact & Organizational Scope: [Yes/No]
Stakeholder Interaction & Influence: [Yes/No]
Experience: [Yes/No]
Supervisory Responsibility: [Yes/No]
Budget & Resource Accountability: [Yes/No]
Working Conditions: [Yes/No]
"""

# Phase 2: Detailed scoring prompt with rubrics
PHASE2_PROMPT = """You are an expert Compensation Analyst. Evaluate the following job description and assign a score of 1-5 for each of the 9 compensable factors using the detailed rubrics below.

SCORING RUBRICS:

### 1. Knowledge & Skills
| Score | Criteria | Example Roles |
|-------|----------|---------------|
| 1 | Basic procedural knowledge, follows standard instructions | File clerk, data entry |
| 2 | Working knowledge, applies standard methods with some judgment | Administrative assistant |
| 3 | Solid proficiency, adapts techniques to varying situations | Analyst, technician |
| 4 | Advanced expertise, develops new solutions and approaches | Senior engineer, specialist |
| 5 | Deep domain mastery, recognized expert, shapes industry practices | Chief architect, director |

### 2. Problem Solving & Complexity
| Score | Criteria | Example Roles |
|-------|----------|---------------|
| 1 | Routine problems with clear, established solutions | Receptionist, cashier |
| 2 | Some variation, uses established guidelines to resolve issues | Customer service rep |
| 3 | Moderate complexity, evaluates multiple options | Project coordinator |
| 4 | Complex analysis, develops novel approaches | Systems architect |
| 5 | Strategic problems, navigates highly ambiguous situations | Executive leadership |

### 3. Decision Authority
| Score | Criteria | Example Roles |
|-------|----------|---------------|
| 1 | No independent decisions, follows explicit instructions | Intern, trainee |
| 2 | Minor decisions within strict guidelines | Clerk, assistant |
| 3 | Moderate autonomy within defined scope | Supervisor |
| 4 | Significant discretion, interprets policy | Manager |
| 5 | Full authority, sets organizational policy | Director, VP, C-suite |

### 4. Impact & Organizational Scope
| Score | Criteria | Example Roles |
|-------|----------|---------------|
| 1 | Individual tasks only, no broader impact | Entry-level |
| 2 | Affects immediate team | Team member |
| 3 | Department-wide impact | Department lead |
| 4 | Multi-department or division impact | Division manager |
| 5 | Enterprise-wide, affects external stakeholders | C-suite, VP |

### 5. Stakeholder Interaction & Influence
| Score | Criteria | Example Roles |
|-------|----------|---------------|
| 1 | Minimal interaction, routine exchanges | Back-office roles |
| 2 | Regular internal communication | Coordinator |
| 3 | Cross-functional collaboration | Project manager |
| 4 | External partners, negotiation required | Account executive |
| 5 | Executive relationships, strategic influence | VP, Director |

### 6. Experience
| Score | Criteria | Example Roles |
|-------|----------|---------------|
| 1 | No prior experience required | Entry-level, intern |
| 2 | 1-2 years relevant experience | Junior roles |
| 3 | 3-5 years, demonstrated competence | Mid-level |
| 4 | 6-10 years, proven track record | Senior roles |
| 5 | 10+ years, industry recognition | Executive, expert |

### 7. Supervisory Responsibility
| Score | Criteria | Example Roles |
|-------|----------|---------------|
| 1 | No supervisory duties | Individual contributor |
| 2 | May train or guide others informally | Senior IC |
| 3 | Supervises small team (1-5 direct reports) | Team lead |
| 4 | Manages multiple teams (6-20 reports) | Manager |
| 5 | Directs large organization (20+ reports) | Director, VP |

### 8. Budget & Resource Accountability
| Score | Criteria | Example Roles |
|-------|----------|---------------|
| 1 | No budget responsibility | Staff |
| 2 | Minor purchasing authority | Coordinator |
| 3 | Manages project-level budgets | Project manager |
| 4 | Department budget oversight | Department manager |
| 5 | Organization-wide financial authority | CFO, Director |

### 9. Working Conditions
| Score | Criteria | Example Roles |
|-------|----------|---------------|
| 1 | Standard office environment | Office worker |
| 2 | Occasional travel or extended hours | Sales rep |
| 3 | Regular field work or physical activity | Inspector |
| 4 | Challenging conditions, exposure to hazards | Field engineer |
| 5 | Hazardous or extreme environments | Emergency responder |

{pre_prompt}

Job Description:
{job_description}

{post_prompt}

INSTRUCTIONS:
- Assign a score from 1-5 for each factor based on the rubrics above.
- If information is not explicitly mentioned, infer from job title, seniority level, or related context.
- If truly unable to determine, default to 3 (moderate).
- Respond ONLY in the exact format below, one factor per line. No explanations.

Knowledge & Skills: [1-5]
Problem Solving & Complexity: [1-5]
Decision Authority: [1-5]
Impact & Organizational Scope: [1-5]
Stakeholder Interaction & Influence: [1-5]
Experience: [1-5]
Supervisory Responsibility: [1-5]
Budget & Resource Accountability: [1-5]
Working Conditions: [1-5]
"""
