from rest_framework import serializers
from .models import YouTubeSummary,PDFSummary,PDFMindMap,GeneratedQuiz,Video

class YouTubeSummarySerializer(serializers.ModelSerializer):
    class Meta:
        model = YouTubeSummary
        fields = ['id', 'user', 'youtube_url', 'transcript', 'summary', 'created_at']
        read_only_fields = ['id', 'user', 'transcript', 'summary', 'created_at']

    # Validate the YouTube URL
    def validate_youtube_url(self, value):
        if "youtube.com" not in value and "youtu.be" not in value:
            raise serializers.ValidationError("Invalid YouTube URL.")
        return value


class PDFSummarySerializer(serializers.ModelSerializer):
    class Meta:
        model = PDFSummary
        fields = ['user', 'pdf_file', 'start_page_number', 'end_page_number', 'summary']

    def create(self, validated_data):
        # Automatically assign the user from the request
        validated_data['user'] = self.context['request'].user
        return super().create(validated_data)




class PDFMindMapSerializer(serializers.ModelSerializer):
    class Meta:
        model = PDFMindMap
        fields = ['id', 'user', 'pdf_file', 'start_page_number', 'end_page_number', 'mindmap_json', 'created_at']
        
        
class GeneratedQuizSerializer(serializers.ModelSerializer):
    class Meta:
        model = GeneratedQuiz
        fields = [ 'user', 'pdf_summary', 'quiz_data', 'created_at']

class VideoSerializer(serializers.ModelSerializer):
    class Meta:
        model = Video
        fields = ['video_file', 'audio_file']  # Include audio_file in the fields
