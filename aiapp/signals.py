from django.db.models.signals import post_save, post_delete
from django.dispatch import receiver
from .models import Comment, Post
import logging

logger = logging.getLogger(__name__)

@receiver([post_save, post_delete], sender=Comment)
def update_comments_count(sender, instance, **kwargs):
    try:
        post = instance.post
        post.comments_count = Comment.objects.filter(post=post).count()
        post.save(update_fields=['comments_count'])
        logger.info(f"Updated comments_count for post_id={post.id}: {post.comments_count}")
    except Exception as e:
        logger.error(f"Error updating comments_count for post_id={instance.post.id}: {str(e)}")