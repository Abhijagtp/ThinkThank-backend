from django.contrib.auth.models import AbstractUser
from django.db import models
from django.utils import timezone

class CustomUser(AbstractUser):
    email = models.EmailField(unique=True)
    company_name = models.CharField(max_length=255, blank=True)
    is_active = models.BooleanField(default=True)
    avatar = models.ImageField(upload_to='avatars/', blank=True, null=True)

    USERNAME_FIELD = 'email'  # Use email for authentication
    REQUIRED_FIELDS = ['username', 'company_name']  # Required for createsuperuser

    def __str__(self):
        return self.email


class Document(models.Model):
    user = models.ForeignKey(CustomUser, on_delete=models.CASCADE, related_name='documents')
    file = models.FileField(upload_to='documents/%Y/%m/%d/')
    name = models.CharField(max_length=255)
    size = models.BigIntegerField()  # File size in bytes
    uploaded_at = models.DateTimeField(auto_now_add=True)
    file_type = models.CharField(max_length=50)  # e.g., 'pdf', 'docx'

    class Meta:
        ordering = ['-uploaded_at']

    def __str__(self):
        return f"{self.name} ({self.user.email})"
    


class ChatHistory(models.Model):
    user = models.ForeignKey(CustomUser, on_delete=models.CASCADE)
    document = models.ForeignKey(Document, on_delete=models.CASCADE, null=True)
    message = models.JSONField()  # Store message as JSON: {id, type, content, timestamp, insights, is_json}
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['created_at']

    def __str__(self):
        return f"Chat for {self.user.email} on {self.document.name if self.document else 'no document'}"


class ComparisonHistory(models.Model):
    user = models.ForeignKey(CustomUser, on_delete=models.CASCADE)
    document1 = models.ForeignKey(Document, on_delete=models.CASCADE, related_name='comparisons_as_doc1')
    document2 = models.ForeignKey(Document, on_delete=models.CASCADE, related_name='comparisons_as_doc2')
    result = models.JSONField()  # Store comparison result: {summary, keyDifferences, insights, recommendations}
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['created_at']

    def __str__(self):
        return f"Comparison between {self.document1.name} and {self.document2.name} for {self.user.email}"


class SavedNote(models.Model):
    user = models.ForeignKey(CustomUser, on_delete=models.CASCADE, related_name='saved_notes')
    title = models.CharField(max_length=255)
    content = models.TextField()
    tags = models.JSONField(default=list)  # e.g., ["market-research", "growth"]
    created_at = models.DateTimeField(default=timezone.now)
    last_modified = models.DateTimeField(auto_now=True)
    source_document = models.ForeignKey('Document', on_delete=models.SET_NULL, null=True, blank=True)
    source_type = models.CharField(max_length=50, choices=[('analysis', 'Analysis'), ('comparison', 'Comparison')], null=True)
    source_id = models.CharField(max_length=100, null=True, blank=True)  # e.g., chat message ID or comparison ID
    starred = models.BooleanField(default=False)
    color = models.CharField(max_length=50, default='blue')  # e.g., blue, purple, green

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f"{self.title} ({self.user.email})"



class Post(models.Model):
    POST_TYPES = [
        ('insight', 'Insight'),
        ('question', 'Question'),
        ('ai', 'AI Highlight'),
    ]
    user = models.ForeignKey(CustomUser, on_delete=models.CASCADE, related_name='posts')
    post_type = models.CharField(max_length=20, choices=POST_TYPES)
    summary = models.TextField(blank=True)  # For insights
    question = models.TextField(blank=True)  # For questions
    bullets = models.JSONField(default=list, blank=True)  # For AI highlights
    tags = models.JSONField(default=list)  # e.g., ["SaaS", "Security"]
    likes = models.PositiveIntegerField(default=0)
    comments_count = models.PositiveIntegerField(default=0)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f"{self.post_type.capitalize()} by {self.user.email} at {self.created_at}"

class PostInteraction(models.Model):
    user = models.ForeignKey(CustomUser, on_delete=models.CASCADE, related_name='post_interactions')
    post = models.ForeignKey(Post, on_delete=models.CASCADE, related_name='interactions')
    liked = models.BooleanField(default=False)
    saved = models.BooleanField(default=False)

    class Meta:
        unique_together = ('user', 'post')  # Ensure one interaction per user per post

    def __str__(self):
        return f"Interaction by {self.user.email} on post {self.post.id}"

class Comment(models.Model):
    user = models.ForeignKey(CustomUser, on_delete=models.CASCADE, related_name='comments')
    post = models.ForeignKey(Post, on_delete=models.CASCADE, related_name='comments')
    content = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['created_at']

    def __str__(self):
        return f"Comment by {self.user.email} on post {self.post.id}"