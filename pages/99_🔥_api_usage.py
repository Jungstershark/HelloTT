import re
from pathlib import Path

import pandas as pd
import streamlit as st
from dateutil import parser

from utils.llm import LOG_FILEPATH

st.title("Azure OpenAI API Usage")

if not Path(LOG_FILEPATH).is_file():
    st.error(f"API usage log file ({LOG_FILEPATH}) not found!")
    st.stop()


# example a single line of usage log
# 2024-02-17 13:43:11.013 | DEBUG    | utils.tokens:log_usage:159 - Prompt Tokens: 8, Prompt Cost (USD): 0.000024, Completion Tokens: 9, Completion Cost (USD): 0.000036, Total Tokens: 17, Total Cost (USD): $0.000060

with open(LOG_FILEPATH) as f:
    raw_data = [
        re.split(" - ", line.strip())
        for line in f.readlines()
        if "Completion Cost (USD): $" in line  # quick hack to filter only usage logs
    ]

usages_records = []
for metadata, log_msg in raw_data:
    usage_data = log_msg.split(",")
    usage_record = {
        "datetime": parser.parse(metadata.split("|")[0]),
        "prompt_tokens": int(usage_data[0].split(":")[1]),
        "prompt_costs_usd": float(usage_data[1].split(":")[1].strip("$ ")),
        "completion_tokens": int(usage_data[2].split(":")[1]),
        "completion_costs_usd": float(usage_data[3].split(":")[1].strip("$ ")),
        "total_tokens": int(usage_data[4].split(":")[1]),
        "total_costs_usd": float(usage_data[5].split(":")[1].strip("$ ")),
    }
    usages_records.append(usage_record)

usage_df = pd.DataFrame(usages_records)

if not usage_df.empty:
    usage_df = usage_df.sort_values("datetime", ascending=False)
    st.dataframe(
        usage_df.style.format(
            {
                "datetime": "{:%Y-%m-%d %I:%M:%S%p}",
                "prompt_costs_usd": "{:.6f}",
                "completion_costs_usd": "{:.6f}",
                "total_costs_usd": "{:.6f}",
            }
        )
    )

    st.subheader(f"All time total tokens: {usage_df['total_tokens'].sum():,}")
    st.bar_chart(usage_df, x="datetime", y="total_tokens")

    st.subheader(f"All time total cost (USD): ${usage_df['total_costs_usd'].sum():.6f}")
    st.bar_chart(usage_df, x="datetime", y="total_costs_usd")
