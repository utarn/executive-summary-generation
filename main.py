import json

import pandas as pd
import numpy as np
import re
from collections import defaultdict
import io
import os

from fastapi import FastAPI, HTTPException
import base64
from fastapi.responses import JSONResponse
from dotenv import load_dotenv

from google import genai
from google.genai import types
from pydantic import BaseModel
import logging

# Load environment variables (for GEMINI_API_KEY)
load_dotenv()

logger = logging.getLogger(__name__)

app = FastAPI()

# --- Hardcoded Province to Region Mapping (same as your original) ---
province_to_region = {
    # ภาคเหนือ (Northern Region)
    'เชียงใหม่': 'ภาคเหนือ', 'เชียงราย': 'ภาคเหนือ', 'ลำปาง': 'ภาคเหนือ', 'ลำพูน': 'ภาคเหนือ',
    'แม่ฮ่องสอน': 'ภาคเหนือ', 'น่าน': 'ภาคเหนือ', 'พะเยา': 'ภาคเหนือ', 'แพร่': 'ภาคเหนือ',
    'อุตรดิตถ์': 'ภาคเหนือ',
    # ภาคตะวันออกเฉียงเหนือ (Northeastern Region - Isan)
    'นครราชสีมา': 'ภาคตะวันออกเฉียงเหนือ', 'อุบลราชธานี': 'ภาคตะวันออกเฉียงเหนือ',
    'ขอนแก่น': 'ภาคตะวันออกเฉียงเหนือ', 'บุรีรัมย์': 'ภาคตะวันออกเฉียงเหนือ',
    'อุดรธานี': 'ภาคตะวันออกเฉียงเหนือ', 'ศรีสะเกษ': 'ภาคตะวันออกเฉียงเหนือ',
    'สุรินทร์': 'ภาคตะวันออกเฉียงเหนือ', 'ร้อยเอ็ด': 'ภาคตะวันออกเฉียงเหนือ',
    'เลย': 'ภาคตะวันออกเฉียงเหนือ', 'ชัยภูมิ': 'ภาคตะวันออกเฉียงเหนือ',
    'สกลนคร': 'ภาคตะวันออกเฉียงเหนือ', 'กาฬสินธุ์': 'ภาคตะวันออกเฉียงเหนือ',
    'มหาสารคาม': 'ภาคตะวันออกเฉียงเหนือ', 'นครพนม': 'ภาคตะวันออกเฉียงเหนือ',
    'มุกดาหาร': 'ภาคตะวันออกเฉียงเหนือ', 'ยโสธร': 'ภาคตะวันออกเฉียงเหนือ',
    'หนองคาย': 'ภาคตะวันออกเฉียงเหนือ', 'หนองบัวลำภู': 'ภาคตะวันออกเฉียงเหนือ',
    'บึงกาฬ': 'ภาคตะวันออกเฉียงเหนือ', 'อำนาจเจริญ': 'ภาคตะวันออกเฉียงเหนือ',
    # ภาคกลาง (Central Region)
    'กรุงเทพมหานคร': 'ภาคกลาง', 'นนทบุรี': 'ภาคกลาง', 'ปทุมธานี': 'ภาคกลาง',
    'สมุทรปราการ': 'ภาคกลาง', 'นครปฐม': 'ภาคกลาง', 'สมุทรสาคร': 'ภาคกลาง',
    'พระนครศรีอยุธยา': 'ภาคกลาง', 'สระบุรี': 'ภาคกลาง', 'ลพบุรี': 'ภาคกลาง',
    'สิงห์บุรี': 'ภาคกลาง', 'ชัยนาท': 'ภาคกลาง', 'อ่างทอง': 'ภาคกลาง',
    'นครสวรรค์': 'ภาคกลาง', 'อุทัยธานี': 'ภาคกลาง', 'กำแพงเพชร': 'ภาคกลาง',
    'พิจิตร': 'ภาคกลาง', 'พิษณุโลก': 'ภาคกลาง', 'สุโขทัย': 'ภาคกลาง',
    'เพชรบูรณ์': 'ภาคกลาง', 'สุพรรณบุรี': 'ภาคกลาง', 'ราชบุรี': 'ภาคกลาง',
    'กาญจนบุรี': 'ภาคกลาง', 'นครนายก': 'ภาคกลาง',
    # ภาคตะวันออก (Eastern Region)
    'ชลบุรี': 'ภาคตะวันออก', 'ระยอง': 'ภาคตะวันออก', 'จันทบุรี': 'ภาคตะวันออก',
    'ตราด': 'ภาคตะวันออก', 'ฉะเชิงเทรา': 'ภาคตะวันออก', 'ปราจีนบุรี': 'ภาคตะวันออก',
    'สระแก้ว': 'ภาคตะวันออก',
    # ภาคตะวันตก (Western Region)
    'ตาก': 'ภาคตะวันตก', 'เพชรบุรี': 'ภาคตะวันตก', 'ประจวบคีรีขันธ์': 'ภาคตะวันตก',
    # ภาคใต้ (Southern Region)
    'สุราษฎร์ธานี': 'ภาคใต้', 'นครศรีธรรมราช': 'ภาคใต้', 'สงขลา': 'ภาคใต้',
    'ภูเก็ต': 'ภาคใต้', 'ตรัง': 'ภาคใต้', 'พัทลุง': 'ภาคใต้', 'กระบี่': 'ภาคใต้',
    'ชุมพร': 'ภาคใต้', 'ระนอง': 'ภาคใต้', 'สตูล': 'ภาคใต้', 'พังงา': 'ภาคใต้',
    'ยะลา': 'ภาคใต้', 'ปัตตานี': 'ภาคใต้', 'นราธิวาส': 'ภาคใต้',
    'อื่นๆ': 'ไม่ระบุภาค'
}

def clean_option_text(option_string):
    if isinstance(option_string, str):
        match = re.search(r'\)\s*(.*)', option_string)
        if match:
            return match.group(1).strip()
        return option_string.strip()
    elif pd.isna(option_string):
        return "ไม่ตอบ / ว่างเปล่า"
    return str(option_string).strip()

def process_excel_data(file_contents: bytes) -> str:
    """
    Processes the Excel file contents and generates a textual report summary.
    """
    try:
        df = pd.read_excel(io.BytesIO(file_contents))
    except Exception as e:
        raise ValueError(f"Error reading Excel file: {e}")

    report_text = ""
    processed_ages = [] # To collect all valid positive ages for overall average and age group analysis
    region_province_map = defaultdict(set) # To store unique provinces for each region

    # --- Data Processing and Analysis ---
    for col_idx, col_name in enumerate(df.columns):
        if col_idx < 2: # Skip 'ลำดับ', 'ประทับเวลา' (assuming these are first two columns)
            continue

        question = col_name
        report_text += f"Question: {question}\n"

        if question == 'อายุ': # Specific handling for 'Age' column
            ages_for_this_column_report = []
            for age_val_raw in df[col_name]:
                cleaned_age_str = clean_option_text(str(age_val_raw))
                try:
                    numeric_age = pd.to_numeric(cleaned_age_str, errors='coerce')
                    ages_for_this_column_report.append(numeric_age) # Include NaN for distribution report
                    if pd.notna(numeric_age) and numeric_age > 0:
                        processed_ages.append(numeric_age) # For average and age groups
                except ValueError:
                    ages_for_this_column_report.append(np.nan)

            if ages_for_this_column_report:
                age_series = pd.Series(ages_for_this_column_report)
                value_percentages = age_series.value_counts(normalize=True, dropna=False).rename(index={np.nan: "ไม่สามารถประมวลผลได้/ไม่ตอบ"}) * 100
                for age_display_val, percentage in value_percentages.items():
                    if isinstance(age_display_val, (int, float)) and pd.notna(age_display_val):
                        report_text += f"- {int(age_display_val)} ปี: {percentage:.1f}%\n"
                    else:
                        report_text += f"- {age_display_val}: {percentage:.1f}%\n"
            else:
                report_text += "- ไม่มีข้อมูลอายุที่สามารถประมวลผลได้\n"

        elif question == 'จังหวัด': # Specific handling for 'Province' column
            regions_list_for_overall_percentage = []
            cleaned_provinces_list_for_individual_report = []

            for province_val_raw in df[col_name]:
                province_text = clean_option_text(province_val_raw)
                cleaned_provinces_list_for_individual_report.append(province_text)

                region = province_to_region.get(province_text, 'ไม่ระบุภาค')
                regions_list_for_overall_percentage.append(region)
                if province_text != "ไม่ตอบ / ว่างเปล่า" and region != 'ไม่ระบุภาค':
                    region_province_map[region].add(province_text)

            if regions_list_for_overall_percentage:
                report_text += "การกระจายตัวตามภูมิภาค:\n"
                region_series = pd.Series(regions_list_for_overall_percentage)
                region_percentages = region_series.value_counts(normalize=True, dropna=False) * 100
                for region_name_from_series, percentage in region_percentages.items():
                    report_text += f"- {region_name_from_series}: {percentage:.1f}%\n"
                    if region_name_from_series in region_province_map and region_province_map[region_name_from_series]:
                        provinces_in_region_str = ", ".join(sorted(list(region_province_map[region_name_from_series])))
                        report_text += f"  (จังหวัดที่พบ: {provinces_in_region_str})\n"
                    elif region_name_from_series != 'ไม่ระบุภาค' and region_name_from_series != 'ไม่ตอบ / ว่างเปล่า':
                         report_text += f"  (ไม่มีข้อมูลจังหวัดสำหรับภาคนี้ หรือ จังหวัดไม่ถูกระบุในแผนที่)\n"
                report_text += "\n"
            else:
                report_text += "ไม่มีข้อมูลภูมิภาคที่สามารถประมวลผลได้\n\n"

            if cleaned_provinces_list_for_individual_report:
                report_text += "การกระจายตัวตามรายจังหวัด:\n"
                province_series = pd.Series(cleaned_provinces_list_for_individual_report)
                province_counts = province_series.value_counts(dropna=False)
                province_percentages = province_series.value_counts(normalize=True, dropna=False) * 100
                for i in range(len(province_counts)):
                    prov_name = province_counts.index[i]
                    count = province_counts.iloc[i]
                    percent = province_percentages.iloc[i]
                    report_text += f"- {prov_name}: {percent:.1f}% (n={count})\n"
            else:
                report_text += "ไม่มีข้อมูลจังหวัดที่สามารถประมวลผลได้\n"
        else: # General handling for other columns
            cleaned_options_series = df[col_name].apply(clean_option_text)
            value_percentages = cleaned_options_series.value_counts(normalize=True, dropna=False) * 100
            if value_percentages.empty:
                report_text += "- ไม่มีข้อมูลสำหรับคำถามนี้\n"
            else:
                for option_text, percentage in value_percentages.items():
                    report_text += f"- {option_text}: {percentage:.1f}%\n"
        report_text += "----\n\n"

    # --- Additional Analysis: Age Groups ---
    report_text += "ข้อมูลเพิ่มเติม - การกระจายตัวตามกลุ่มอายุ:\n"
    if processed_ages:
        age_group_15_29 = len([age for age in processed_ages if 15 <= age <= 29])
        age_group_30_49 = len([age for age in processed_ages if 30 <= age <= 49])
        age_group_50_plus = len([age for age in processed_ages if age >= 50])
        total_valid_ages_for_grouping = age_group_15_29 + age_group_30_49 + age_group_50_plus

        if total_valid_ages_for_grouping > 0:
            percent_15_29 = (age_group_15_29 / total_valid_ages_for_grouping) * 100
            percent_30_49 = (age_group_30_49 / total_valid_ages_for_grouping) * 100
            percent_50_plus = (age_group_50_plus / total_valid_ages_for_grouping) * 100
            report_text += f"- กลุ่มอายุ 15-29 ปี: {percent_15_29:.1f}% (n={age_group_15_29})\n"
            report_text += f"- กลุ่มอายุ 30-49 ปี: {percent_30_49:.1f}% (n={age_group_30_49})\n"
            report_text += f"- กลุ่มอายุ 50 ปีขึ้นไป: {percent_50_plus:.1f}% (n={age_group_50_plus})\n"
        else:
            report_text += "- ไม่มีข้อมูลอายุที่สามารถจัดกลุ่มได้ (15 ปีขึ้นไป)\n"
    else:
        report_text += "- ไม่มีข้อมูลอายุที่ประมวลผลได้สำหรับการจัดกลุ่ม\n"
    report_text += "----\n\n"

    # --- Calculate and Append Average Age ---
    report_text += "ข้อมูลเพิ่มเติม - อายุเฉลี่ย:\n"
    if processed_ages:
        average_age = np.mean(processed_ages)
        report_text += f"- อายุเฉลี่ยของผู้ตอบแบบสอบถาม (จากข้อมูลที่ประมวลผลได้และมากกว่า 0): {average_age:.1f} ปี\n"
    else:
        report_text += f"- ไม่สามารถคำนวณอายุเฉลี่ยได้เนื่องจากไม่มีข้อมูลอายุที่ถูกต้อง (มากกว่า 0)\n"
    report_text += "----\n\n"

    logger.info("Report summary generated successfully.")
    logger.debug(f"Report summary: {report_text}")
    return report_text


def generate_executive_summary_with_gemini(report_summary_text: str) -> str:
    """
    Generates an executive summary using Gemini API (non-streaming).
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable not set.")

    client = genai.Client(
        api_key=os.environ.get("GEMINI_API_KEY"),
    )
    model = "gemini-2.5-pro-preview-06-05"

    prompt = f"""{report_summary_text}
####
Each group will have question and the set of answer with its percentage.
Note that in some question you may find 0.0% as it may have fewer respondents that 0.01% or it may be an open question that the user response, just do not ignore those answer though it has 0.0%.
Please write an executive summary in Thai summary to try to identity the intention from the set of questions and explain each questions and choices briefly within 500 words.
Do no write about invalid responses and data."""

    try:
        response = client.models.generate_content(
            model=model,
            contents=[
                types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=prompt)],
                )
            ],
            config=types.GenerateContentConfig(
                response_mime_type="text/plain",
                temperature=1,
            )
        )

        if response.text:
            result = {
                "text": response.text,
                "usage" : {
                    "prompt_tokens": response.usage_metadata.prompt_token_count,
                    "candidates_tokens": response.usage_metadata.candidates_token_count,
                    "total_tokens": response.usage_metadata.total_token_count,

                }
            }

            json_response = json.dumps(result, ensure_ascii=False)
            return json_response

        else:
            # Handle cases where response might be empty or blocked
            safety_feedback = response.prompt_feedback if response.prompt_feedback else "No specific safety feedback."
            return f"Gemini API did not return content. Possible safety block or empty response. Feedback: {safety_feedback}"

    except Exception as e:
        raise RuntimeError(f"Error calling Gemini API: {e}")


class SummaryRequest(BaseModel):
    uploaded_file: str  # base64-encoded file content

class SummaryResponse(BaseModel):
    summary: str

@app.post("/generate-summary", response_model=SummaryResponse)
async def create_summary_from_base64(request: SummaryRequest):
    """
    Accepts a base64-encoded Excel file in JSON, processes it, generates a report summary,
    then uses Gemini to create an executive summary, and returns it as JSON.
    """
    try:
        file_contents = base64.b64decode(request.uploaded_file)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base64-encoded file.")

    # Optionally, check file signature for Excel (magic bytes) or try reading to validate
    try:
        report_summary = process_excel_data(file_contents)
    except ValueError as e: # Catch specific error from process_excel_data
        raise HTTPException(status_code=400, detail=f"Error processing Excel data: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred during data processing: {str(e)}")

    if not report_summary:
        raise HTTPException(status_code=500, detail="Failed to generate report summary from Excel data.")

    try:
        executive_summary = generate_executive_summary_with_gemini(report_summary)
    except ValueError as e: # Catch specific error from Gemini function (e.g. API key missing)
        raise HTTPException(status_code=500, detail=str(e))
    except RuntimeError as e: # Catch specific error from Gemini API call
        raise HTTPException(status_code=502, detail=f"Error communicating with Gemini API: {str(e)}") # 502 Bad Gateway
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred during executive summary generation: {str(e)}")

    return JSONResponse(content=executive_summary)

if __name__ == "__main__":
    import uvicorn
    # It's good practice to get host and port from environment variables too for deployment
    host = os.getenv("APP_HOST", "127.0.0.1")
    port = int(os.getenv("APP_PORT", "8001"))
    print(f"Starting Uvicorn server on {host}:{port}")
    uvicorn.run(app, host=host, port=port)