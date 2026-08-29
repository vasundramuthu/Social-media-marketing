from textblob import TextBlob

class SentimentEngine:
    """
    A utility class to process qualitative data for InfluenceIQ.
    This helps transform raw text comments into numerical safety scores.
    """
    
    @staticmethod
    def calculate_safety_score(comments):
        """
        Processes a list of strings and returns a dictionary with 
        sentiment metrics and a safety recommendation.
        """
        if not comments or len(comments) == 0:
            return {
                "score": 50,
                "label": "Neutral/Unknown",
                "color": "#94a3b8", # Slate-400
                "sentiment_polarity": 0
            }

        # Analyze each comment
        polarities = []
        for comment in comments:
            analysis = TextBlob(comment)
            polarities.append(analysis.sentiment.polarity)

        # Average polarity ranges from -1.0 to 1.0
        avg_polarity = sum(polarities) / len(polarities)
        
        # Normalize to 0-100 scale for the UI
        # -1 becomes 0, 0 becomes 50, 1 becomes 100
        display_score = round((avg_polarity + 1) * 50, 1)

        # Determine Category
        if display_score > 75:
            label, color = "Excellent", "#10b981" # Emerald-500
        elif display_score > 55:
            label, color = "Safe", "#3b82f6"    # Blue-500
        elif display_score > 45:
            label, color = "Neutral", "#f59e0b" # Amber-500
        else:
            label, color = "Risky", "#ef4444"   # Red-500

        return {
            "score": display_score,
            "label": label,
            "color": color,
            "sentiment_raw": round(avg_polarity, 3),
            "comment_count": len(comments)
        }