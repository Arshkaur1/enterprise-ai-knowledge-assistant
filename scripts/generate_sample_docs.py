
from pathlib import Path

import pymupdf
from docx import Document
from docx.shared import Pt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)


# -------------------------------------------------
# 1. HR POLICY - TXT
# -------------------------------------------------

hr_content = """
NORTHSTAR TECHNOLOGIES
EMPLOYEE HUMAN RESOURCES POLICY

1. REMOTE WORK POLICY

Employees may work remotely up to three days per week.
Employees must obtain approval from their direct manager
before adopting a recurring remote work arrangement.

Employees working remotely must remain available during
their agreed working hours. Sensitive company information
must only be accessed using approved company devices.

2. ANNUAL LEAVE POLICY

Full-time employees receive 20 working days of annual
paid leave per calendar year.

Leave requests must be submitted through the employee
portal at least five working days before the intended
leave date. Manager approval is required.

3. SICK LEAVE POLICY

Employees may take sick leave when they are unable to
perform their duties because of illness.

Employees should notify their manager as soon as
reasonably possible. Medical documentation may be
requested for extended absences.

4. EMPLOYEE BENEFITS

Eligible full-time employees receive health coverage
and access to professional development programs.

Employees should contact Human Resources for benefit
eligibility and enrollment information.
"""

(RAW_DATA_DIR / "hr_policy.txt").write_text(
    hr_content.strip(),
    encoding="utf-8",
)


# -------------------------------------------------
# 2. IT SECURITY POLICY - DOCX
# -------------------------------------------------

doc = Document()

doc.add_heading(
    "NORTHSTAR TECHNOLOGIES",
    level=0,
)

doc.add_heading(
    "IT SECURITY AND ACCESS CONTROL POLICY",
    level=1,
)

sections = [
    (
        "1. PASSWORD REQUIREMENTS",
        "All employees must use unique passwords for company "
        "accounts. Passwords must contain at least 12 "
        "characters and include a combination of uppercase "
        "letters, lowercase letters, numbers and symbols. "
        "Multi-factor authentication is mandatory."
    ),
    (
        "2. ACCESS CONTROL",
        "Employees receive access only to systems and data "
        "required for their assigned responsibilities. "
        "Managers must approve access requests. Access "
        "permissions must be reviewed periodically."
    ),
    (
        "3. SECURITY INCIDENT REPORTING",
        "Employees must report suspected phishing attacks, "
        "unauthorized access and potential data breaches "
        "to the IT Security team immediately. Incident "
        "reports must include the approximate time, affected "
        "systems and relevant observations."
    ),
    (
        "4. DEVICE SECURITY",
        "Company devices must use approved security software. "
        "Employees must enable automatic security updates "
        "and lock their devices when unattended. Company "
        "data must not be transferred to unauthorized "
        "personal storage services."
    ),
]

for heading, content in sections:
    doc.add_heading(heading, level=2)
    paragraph = doc.add_paragraph(content)
    assert paragraph.style is not None
    paragraph.style.font.size = Pt(11)

doc.save(str(RAW_DATA_DIR / "it_security.docx"))


# -------------------------------------------------
# 3. FINANCE POLICY - PDF
# -------------------------------------------------

pdf_path = RAW_DATA_DIR / "finance_policy.pdf"

finance_sections = [
    (
        "1. BUSINESS EXPENSE ELIGIBILITY",
        "Employees may claim reimbursement for approved "
        "business-related travel, accommodation, meals "
        "and necessary business supplies. Personal expenses "
        "are not eligible for reimbursement."
    ),
    (
        "2. EXPENSE LIMITS",
        "Meal expenses are reimbursable up to 40 currency "
        "units per employee per day. Accommodation expenses "
        "are reimbursable up to 150 currency units per night. "
        "Exceptions require advance approval from Finance."
    ),
    (
        "3. REIMBURSEMENT PROCESS",
        "Employees must submit expense claims through the "
        "finance portal within 30 calendar days of the "
        "transaction. Each claim must include an itemized "
        "receipt and a description of the business purpose."
    ),
    (
        "4. APPROVAL WORKFLOW",
        "Claims below 500 currency units require manager "
        "approval. Claims of 500 currency units or more "
        "require both manager and Finance department "
        "approval. Incomplete claims may be returned."
    ),
]

pdf = pymupdf.open()
page = pdf.new_page()

page.insert_text(
    (60, 60),
    "NORTHSTAR TECHNOLOGIES",
    fontsize=18,
    fontname="helv",
)

page.insert_text(
    (60, 95),
    "FINANCE AND EXPENSE REIMBURSEMENT POLICY",
    fontsize=13,
    fontname="helv",
)

y_position = 145

for heading, content in finance_sections:

    page.insert_text(
        (60, y_position),
        heading,
        fontsize=12,
        fontname="hebo",
    )

    y_position += 22

    lines = [
        content[i:i + 85]
        for i in range(0, len(content), 85)
    ]

    for line in lines:
        page.insert_text(
            (60, y_position),
            line,
            fontsize=10,
            fontname="helv",
        )
        y_position += 16

    y_position += 18

pdf.save(pdf_path)
pdf.close()


print("Synthetic enterprise dataset generated successfully.")

for file_path in RAW_DATA_DIR.iterdir():
    if file_path.is_file():
        print(f"Created: {file_path.name}")
