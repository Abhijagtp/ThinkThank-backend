from rest_framework import serializers
from django.contrib.auth import get_user_model
from .models import Document,ChatHistory,ComparisonHistory,SavedNote,Post,PostInteraction,Comment

User = get_user_model()

class UserSerializer(serializers.ModelSerializer):
    password = serializers.CharField(write_only=True, min_length=8)

    class Meta:
        model = User
        fields = ('id', 'email', 'username', 'company_name', 'password')

    def create(self, validated_data):
        user = User.objects.create_user(
            email=validated_data['email'],
            username=validated_data.get('username', ''),
            password=validated_data['password'],
            company_name=validated_data.get('company_name', '')
        )
        return user

class LoginSerializer(serializers.Serializer):
    email = serializers.EmailField()
    password = serializers.CharField(write_only=True)


class DocumentSerializer(serializers.ModelSerializer):
    file_url = serializers.SerializerMethodField()

    class Meta:
        model = Document
        fields = ['id', 'name', 'size', 'file_type', 'uploaded_at', 'file', 'file_url']
        read_only_fields = ['id', 'uploaded_at', 'file_type', 'size']

    def get_file_url(self, obj):
        request = self.context.get('request')
        if obj.file and hasattr(obj.file, 'url') and request:
            return request.build_absolute_uri(obj.file.url)
        return None  # Return None if file or request is unavailable

    def create(self, validated_data):
        validated_data['user'] = self.context['request'].user
        file = validated_data['file']
        validated_data['file_type'] = file.name.split('.')[-1].lower()
        validated_data['size'] = file.size
        return super().create(validated_data)
    


class ChatHistorySerializer(serializers.ModelSerializer):
    class Meta:
        model = ChatHistory
        fields = ['message', 'created_at']


class ComparisonHistorySerializer(serializers.ModelSerializer):
    document1 = DocumentSerializer()
    document2 = DocumentSerializer()
    class Meta:
        model = ComparisonHistory
        fields = ['id', 'document1', 'document2', 'result', 'created_at']


class SavedNoteSerializer(serializers.ModelSerializer):
    class Meta:
        model = SavedNote
        fields = ['id', 'title', 'content', 'tags', 'created_at', 'last_modified', 'source_document', 'source_type', 'source_id', 'starred', 'color']



class PostSerializer(serializers.ModelSerializer):
    user = UserSerializer(read_only=True)
    is_liked = serializers.SerializerMethodField()
    is_saved = serializers.SerializerMethodField()

    class Meta:
        model = Post
        fields = ['id', 'post_type', 'user', 'summary', 'question', 'bullets', 'tags', 'likes', 'comments_count', 'created_at', 'is_liked', 'is_saved']

    def get_is_liked(self, obj):
        user = self.context['request'].user
        if user.is_authenticated:
            return PostInteraction.objects.filter(post=obj, user=user, liked=True).exists()
        return False

    def get_is_saved(self, obj):
        user = self.context['request'].user
        if user.is_authenticated:
            return PostInteraction.objects.filter(post=obj, user=user, saved=True).exists()
        return False

class CommentSerializer(serializers.ModelSerializer):
    user = UserSerializer(read_only=True)

    class Meta:
        model = Comment
        fields = ['id', 'user', 'post', 'content', 'created_at']
        read_only_fields = ['id', 'user', 'post', 'created_at']
        extra_kwargs = {
            'content': {'required': True, 'allow_blank': False}
        }

    def validate_content(self, value):
        if not value.strip():
            raise serializers.ValidationError("Content cannot be empty.")
        return value