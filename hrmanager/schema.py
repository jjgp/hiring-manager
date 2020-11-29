HIGH_PERFORMER_COL = "High_Performer"
PROTECTED_GROUP_COL = "Protected_Group"
RETAINED_COL = "Retained"
CRITERION_COLS = (
    "Overall_Rating",
    "Technical_Skills",
    "Teamwork",
    "Customer_Service",
    "Hire_Again",
    HIGH_PERFORMER_COL,
    PROTECTED_GROUP_COL,
    RETAINED_COL
)
BINARY_CRITERION_COLS = ("Hire_Again",)
BINARY_PREDICTOR_COLS = tuple(f"SJ_Most_{n}" for n in range(1, 10)) \
    + tuple(f"SJ_Least_{n}" for n in range(1, 10)) \
    + tuple(f"Scenario{i}_{j}" for i in range(1, 3) for j in range(1, 9)) \
    + tuple(f"Biodata_{n:02}" for n in range(1, 21)) \
    + tuple(f"PScale{i:02}_Q{j}" for i in range(1, 14) for j in range(1, 5)) \
    + ("PScale06_Q5", "PScale06_Q6", "PScale13_Q5",)
BINARY_COLS = BINARY_CRITERION_COLS + BINARY_PREDICTOR_COLS
INDEX_COL = "UNIQUE_ID"
NUMERIC_CRITERION_COLS = (
    "Overall_Rating", "Technical_Skills", "Teamwork", "Customer_Service",
)
NUMERIC_PREDICTOR_COLS = ("Scenario1_Time", "Scenario2_Time") \
    + tuple(f"SJ_Time_{n}" for n in range(1, 10))
NUMERIC_COLS = NUMERIC_CRITERION_COLS + NUMERIC_PREDICTOR_COLS
PREDICTOR_COLS = BINARY_PREDICTOR_COLS + NUMERIC_PREDICTOR_COLS
TARGET_COLS = (HIGH_PERFORMER_COL, PROTECTED_GROUP_COL, RETAINED_COL)
