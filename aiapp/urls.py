from django.urls import path
from .views import RegisterView, LoginView, LogoutView,DocumentUploadView,DocumentListView,DocumentAnalysisView,ChatHistoryView,DocumentComparisonView,ComparisonHistoryView,SavedNoteListCreateView,SavedNoteDetailView,PostInteractionView,PostListCreateView,CommentListCreateView,UserCommentsView,UserView
from rest_framework_simplejwt.views import TokenRefreshView

urlpatterns = [
    path('register/', RegisterView.as_view(), name='register'),
    path('login/', LoginView.as_view(), name='login'),
    path('logout/', LogoutView.as_view(), name='logout'),
    path('token/refresh/', TokenRefreshView.as_view(), name='token_refresh'),
    path('documents/upload/', DocumentUploadView.as_view(), name='document_upload'),
    path('documents/', DocumentListView.as_view(), name='document_list'),
    path('analyze/', DocumentAnalysisView.as_view(), name='document_analysis'),
    path('chat-history/<int:document_id>/', ChatHistoryView.as_view(), name='chat_history'),
    path('compare/', DocumentComparisonView.as_view(), name='compare'),
    path('comparison-history/', ComparisonHistoryView.as_view(), name='comparison_history'),
    path('notes/', SavedNoteListCreateView.as_view(), name='saved_notes_list_create'),
    path('notes/<int:note_id>/', SavedNoteDetailView.as_view(), name='saved_note_detail'),
    # Add paths for posts and comments
    path('posts/', PostListCreateView.as_view(), name='post-list-create'),
    path('posts/<int:post_id>/interact/', PostInteractionView.as_view(), name='post-interact'),
    path('posts/<int:post_id>/comments/', CommentListCreateView.as_view(), name='comment-list-create'),
    path('comments/', UserCommentsView.as_view(), name='user-comments'),
    path('users/me/', UserView.as_view(), name='user-profile')
]