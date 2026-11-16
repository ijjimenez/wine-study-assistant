'''Streamlit interface of the wine study assistant.

Run locally with:  streamlit run app.py
'''
import os

import streamlit as st
from groq import RateLimitError

from rag import ChatSession, WineAssistant, load_config

EXAMPLE_QUESTIONS = [
    'What is the minimum ageing for a Rioja Gran Reserva red wine?',
    'Is a German Spätlese always a sweet wine?',
    'How does carbonic maceration affect the style of a wine?',
    'Tell me about Burgundy',
]

st.set_page_config(page_title='Wine Study Assistant', page_icon=':wine_glass:')


def get_groq_key():
    '''Read the API key from Streamlit secrets, falling back to an environment variable.'''
    try:
        return st.secrets['GROQ_API_KEY']
    except Exception:
        return os.environ.get('GROQ_API_KEY')


@st.cache_resource(show_spinner='Loading the knowledge base...')
def get_assistant():
    '''Load the models and the index once per server, shared by every user session.'''
    return WineAssistant(load_config(), groq_api_key=get_groq_key())


def escape_brackets(text):
    '''Show citation numbers literally instead of letting Markdown read them as links.'''
    return text.replace('[', '\\[').replace(']', '\\]')


def render_answer(message):
    '''Display an assistant turn: the answer, its sources or related topics, and the retrieved passages.'''
    st.markdown(message['reply'])
    if message.get('outcome') == 'answered_uncited':
        st.caption('This answer does not cite its sources. Check it against the retrieved passages below.')
    if message['sources']:
        lines = [f'{escape_brackets(numbers)} [{title}]({url})' for numbers, title, url in message['sources']]
        st.markdown('**Sources**  \n' + '  \n'.join(lines))
    if message['related_topics']:
        st.markdown('**Related topics I can help with:** ' + ', '.join(message['related_topics']))
    if message['passages']:
        with st.expander('Retrieved passages'):
            for title, section, score in message['passages']:
                st.markdown(f'- {title} > {section} (similarity {score:.3f})')


assistant = get_assistant()
memory_settings = assistant.config['memory']
if 'session' not in st.session_state:
    st.session_state.session = ChatSession(memory_settings['max_turns'], memory_settings['max_tokens'])
    st.session_state.messages = []  # what the interface shows, one dict per turn

# --- Sidebar: capabilities, limitations and privacy (user guidance) -----------------------
with st.sidebar:
    st.header('About')
    st.markdown(
        'This assistant answers questions on viticulture, winemaking, wine regions, grape varieties, '
        'wine law and food pairing. It relies only on a curated collection of English Wikipedia articles '
        '(CC BY-SA 4.0) and cites the sources of every answer.'
    )
    st.header('Limitations')
    st.markdown(
        '- Answers may contain errors or omissions, so check important facts against your official study materials.\n'
        '- It only knows what its sources contain and says so when a topic is not covered.\n'
        '- It is an independent study aid, not affiliated with or endorsed by WSET or any certification body.\n'
        '- It does not give health or alcohol-consumption advice.'
    )
    st.header('Privacy')
    provider_note = 'a third-party inference provider (Groq)' if assistant.provider == 'groq' else 'a local model'
    st.markdown(
        f'No account or personal data is required. Questions are processed by {provider_note} and stored '
        'anonymously to improve the assistant, so please do not include personal information.'
    )
    if st.button('New conversation'):
        st.session_state.session = ChatSession(memory_settings['max_turns'], memory_settings['max_tokens'])
        st.session_state.messages = []
        st.rerun()

# --- Main area ------------------------------------------------------------------------------
st.title('Wine Study Assistant')
st.caption(
    'An independent study aid for students preparing wine certifications such as WSET Levels 2 and 3. '
    'Answers may contain errors, so verify important facts.'
)

if not st.session_state.messages:
    st.markdown('**Try one of these questions:**')
    columns = st.columns(2)
    for i, example in enumerate(EXAMPLE_QUESTIONS):
        if columns[i % 2].button(example, key=f'example_{i}'):
            st.session_state.pending_question = example

for message in st.session_state.messages:
    with st.chat_message(message['role']):
        if message['role'] == 'user':
            st.markdown(message['content'])
        else:
            render_answer(message)

question = st.chat_input('Ask about grapes, regions, winemaking or wine law...')
if st.session_state.get('pending_question'):
    question = st.session_state.pop('pending_question')

if question:
    st.session_state.messages.append({'role': 'user', 'content': question})
    with st.chat_message('user'):
        st.markdown(question)

    with st.chat_message('assistant'):
        try:
            with st.spinner('Searching my sources...'):
                result = assistant.chat(question, st.session_state.session)
            message = {
                'role': 'assistant',
                'reply': result['reply'],
                'outcome': result['outcome'],
                'sources': result['sources'],
                'related_topics': result['related_topics'],
                'passages': [(row.title, row.section, float(row.score)) for row in result['passages'].itertuples()],
            }
            render_answer(message)
            st.session_state.messages.append(message)
        except RateLimitError:
            st.warning('The free inference quota is momentarily exhausted. Please wait a minute and ask again.')
        except Exception as error:
            print(f'Error while answering: {error!r}', flush=True)
            st.error('Something went wrong while generating the answer. Please try again.')
