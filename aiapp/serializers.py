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
    class Meta:
        model = Document
        fields = ['id', 'name', 'file', 'file_type', 'size', 'uploaded_at', 'user']
        read_only_fields = ['id', 'uploaded_at', 'user', 'size', 'file_type']

    def validate_file(self, value):
        allowed_types = ['pdf', 'docx', 'doc', 'csv', 'xlsx']
        file_type = value.name.split('.')[-1].lower()
        if file_type not in allowed_types:
            raise serializers.ValidationError('Unsupported file type')
        if value.size > 10 * 1024 * 1024:
            raise serializers.ValidationError('File size exceeds 10MB')
        return value

    def create(self, validated_data):
        public_id = validated_data.pop('public_id', None)
        document = Document(**validated_data)
        if public_id:
            document.file.public_id = public_id  # Set public_id before saving
        document.save()
        return document


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