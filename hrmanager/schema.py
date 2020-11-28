CRITERION_COLS = (
    "Overall_Rating",
    "Technical_Skills",
    "Teamwork",
    "Customer_Service",
    "Hire_Again",
    "High_Performer",
    "Protected_Group",
    "Retained"
)
BINARY_COLS = tuple(f"SJ_Most_{n}" for n in range(1, 10)) \
    + tuple(f"SJ_Least_{n}" for n in range(1, 10)) \
    + tuple(f"Scenario{i}_{j}" for i in range(1, 3) for j in range(1, 9)) \
    + tuple(f"Biodata_{n:02}" for n in range(1, 21)) \
    + tuple(f"PScale{i:02}_Q{j}" for i in range(1, 14) for j in range(1, 5)) \
    + ("PScale06_Q5", "PScale06_Q6", "PScale13_Q5",)
NUMERIC_COLS = ("Scenario1_Time", "Scenario2_Time") \
    + tuple(f"SJ_Time_{n}" for n in range(1, 10))
PREDICTOR_COLS = BINARY_COLS + NUMERIC_COLS
