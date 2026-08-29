class CampaignOptimizer:
    @staticmethod
    def get_strategic_advice(metrics, cluster_label):
        """
        Generates text-based strategy based on ML outputs.
        """
        roi = metrics.get('predicted_roi', 0)
        conv = metrics.get('conversion_rate', 0)
        
        advice = []
        
        # Logic for Micro-Influencers with High Conversion
        if cluster_label == "Micro" and conv > 3.0:
            advice.append("💎 High-Trust Micro: Focus on 'Story' takeovers rather than feed posts.")
        
        # Logic for Premium with Negative ROI
        if cluster_label == "Premium" and roi < 0:
            advice.append("⚠️ Vanity Metric Warning: High reach but low return. Negotiate a fixed fee rather than a performance bonus.")

        # Logic for Macro with High Engagement
        if cluster_label == "Macro" and metrics.get('engagement_rate', 0) > 5.0:
            advice.append("🚀 Rising Star: This influencer's community is highly active. Secure a long-term contract before their rates increase.")

        return advice if advice else ["Maintain current monitoring."]