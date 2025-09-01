import google.generativeai as genai
import json
from typing import Optional, Dict


class ExternalAIReviewer:
    def __init__(self, api_key: Optional[str] = None, model: str = "gemini-1.5-flash"):
        if not api_key:
            raise ValueError("Google AI Studio API key required")
        self.api_key = api_key
        self.model = model

        # Configure Google AI SDK
        genai.configure(api_key=self.api_key)
        self.client = genai.GenerativeModel(self.model)

    def analyze_code_change(self, added_code: str, removed_code: str, context: str, file_path: str) -> Dict:
        prompt = f"""
        Analyze this code change and provide specific technical feedback:

        File: {file_path}

        Removed code:
        ```
        {removed_code if removed_code else '[No removed code]'}
        ```

        Added code:
        ```
        {added_code}
        ```

        Context:
        ```
        {context}
        ```

        Please provide:
        1. Technical assessment of the change quality
        2. Potential issues or improvements
        3. Best practice recommendations
        4. Security considerations if applicable
        5. Performance implications

        Respond in strict JSON with keys: assessment, issues, recommendations, security, performance.
        """

        try:
            response = self.client.generate_content(prompt)
            content = response.text.strip()

            # Try to parse JSON, fallback to raw text
            try:
                return json.loads(content)
            except:
                return {"analysis": content}

        except Exception as e:
            return {"error": f"Request failed: {str(e)}"}
