css = '''
<style>
.chat-message {
    padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 1rem; display: flex
}
.chat-message.user {
    background-color: #2b313e
}
.chat-message.bot {
    background-color: #475063
}
.chat-message .avatar {
  width: 20%;
}
.chat-message .avatar img {
  max-width: 78px;
  max-height: 78px;
  border-radius: 50%;
  object-fit: cover;
}
.chat-message .message {
  width: 80%;
  padding: 0 1.5rem;
  color: #fff;
}
'''

bot_template = '''
<div class="chat-message bot">
    <div class="avatar">
        <img src="https://www.slashfilm.com/img/gallery/the-correct-order-to-watch-the-terminator-movies/intro-1705950026.jpg" style="max-height: 78px; max-width: 78px; border-radius: 50%; object-fit: cover;">
    </div>
    <div class="message">{{MSG}}</div>
</div>
'''

user_template = '''
<div class="chat-message user">
    <div class="avatar">
        <img src="https://m.media-amazon.com/images/M/MV5BYWNiN2VkOTMtOGFmZC00NDI1LTkxMDQtMWI0Y2Y2MDZiY2UzXkEyXkFqcGdeQXVyMTUzMTg2ODkz._V1_.jpg">
    </div>    
    <div class="message">{{MSG}}</div>
</div>
'''