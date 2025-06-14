import os
from openai import OpenAI
from dotenv import load_dotenv
import json
import requests
import warnings
warnings.filterwarnings('ignore')
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
#from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI

from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferWindowMemory
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate,PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from typing import Dict, Any, List
#from IPython.display import display, Markdown

class EnhancedNDAAnalyzer:
    def __init__(self, openai_api_key: str, model_name: str = 'gpt-4o'):
        """Initialize the enhanced NDA analyzer"""
        self.llm = ChatOpenAI(
            openai_api_key=openai_api_key,
            model_name=model_name,
            temperature=0.2
        )
        self.embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        self.vectorstore = None
        self.documents = None
        self.pdf_path = None
        self.memory = ConversationBufferWindowMemory(
            k=10,
            return_messages=True,
            memory_key="chat_history"
        )

        # Define all prompts
        self._setup_prompts()

        # Setup intent classifier
        self._setup_intent_classifier()

    def _ensure_string_response(self, response) -> str:
        """Ensure response is always a string, regardless of input type"""
        if isinstance(response, str):
            return response
        elif isinstance(response, dict):
            # Debug: print what we're getting
            print(f"🔧 Debug: Got dict response with keys: {list(response.keys())}")

            # Try common keys that LangChain might use
            for key in ['output_text', 'text', 'content', 'result', 'answer']:
                if key in response:
                    print(f"🔧 Debug: Using key '{key}' from response")
                    return str(response[key])

            # If no common key found, convert entire dict to string
            print("🔧 Debug: No standard key found, converting entire dict")
            return str(response)
        else:
            # For any other type, convert to string
            print(f"🔧 Debug: Converting {type(response)} to string")
            return str(response)

    def _setup_prompts(self):
        """Setup all prompt templates"""

        # 1. Document Summary Prompt
        self.summary_prompt = '''You are a legal assistant specializing in NDA analysis. Provide a clear, structured summary of this NDA document.

**Focus on these key elements:**

1. **Document Overview:** Type of agreement, effective date, parties involved
2. **Key Parties:** Disclosing party, receiving party, any third parties
3. **Purpose:** Why confidential information is being shared
4. **Key Terms:** Duration, governing law, jurisdiction
5. **Notable Features:** Any unusual or standard clauses

**Format:** Use clear headings and bullet points. Keep it concise but comprehensive.

Document: {text}'''

        # 2. Legal Analysis Prompt (the one you provided earlier)
        self.legal_analysis_prompt = '''You are a legal document analyzer specializing in Non-Disclosure Agreements (NDAs) for a private equity firm called Strada. Your task is to thoroughly review NDAs and identify potential issues, missing clauses, and areas that require attention based on the firm's specific requirements.

## Analysis Framework

### CRITICAL EXCLUSIONS (Must NOT be included)
**Flag these items as HIGH PRIORITY issues:**

1. **Broad Liability Clauses**
   - Direct AND indirect damages compensation requirements
   - Any clause imposing excessive liability on Strada

2. **Overly Broad Non-Solicitation**
   - Clauses covering Strada's affiliated entities without distinction
   - Non-solicitation periods exceeding 1 year
   - Missing exemptions for portfolio companies/affiliates
   - Missing exemptions for unsolicited applications and general media advertisements

3. **Information Retention Restrictions**
   - Clauses preventing retention of any confidential information
   - Missing allowances for "secondary information" retention

4. **Penalty Clauses**
   - Any monetary penalty amounts (e.g., €50k per breach)
   - Specific financial penalties for breaches

5. **Non-Compete Clauses**
   - Any restrictions on Strada's business activities

6. **IP Transfer Provisions**
   - Explicit or implicit transfer of intellectual property rights

### MANDATORY INCLUSIONS (Must be included)
**Flag missing items as HIGH PRIORITY:**

1. **Investment Disclaimer**
   - Clear statement that NDA does not constitute investment commitment
   - Language preventing interpretation as obligation to invest

2. **Broad Disclosee Definition**
   - Directors, employees, shareholders
   - Potential syndicate members
   - Financial service providers
   - Financial and professional advisors

3. **Electronic Data Retention Exception**
   - Confidential information on electronic carriers under automatic archiving
   - Data security procedure exceptions for return/destruction requirements

4. **Regulatory Compliance Retention**
   - Right to retain information for judicial compliance
   - Governmental, supervisory, or regulatory order compliance
   - Audit and compliance purposes

### PREFERRED TERMS (Note deviations)
**Flag as MEDIUM PRIORITY:**

1. **Governing Law & Jurisdiction**
   - Preferred: Belgian law and Belgian courts
   - Acceptable: European alternatives
   - Flag: Non-European jurisdictions (note as potential issue)

2. **Term Duration**
   - Ideal: 2 years
   - Acceptable: Up to 3 years
   - Special cases: Healthcare files up to 5 years
   - Flag: Terms exceeding these limits

### RECIPROCITY CHECK
**Verify mutual obligations:**
- Confirm all recipient obligations apply to both parties
- Check that rights and restrictions are balanced
- Flag one-sided provisions

## Output Format

### Executive Summary
- Overall assessment (Acceptable/Needs Revision/Reject)
- Number of critical issues found
- Key concerns summary

### Critical Issues (HIGH PRIORITY)
- List each problematic clause with specific location
- Explain why it's problematic for Strada
- Suggest specific revisions

### Missing Mandatory Provisions (HIGH PRIORITY)
- Identify required clauses not present
- Provide suggested language for each

### Recommended Improvements (MEDIUM PRIORITY)
- Non-critical but preferred changes
- Jurisdiction and term concerns
- Reciprocity issues

### Acceptable Provisions
- Highlight well-drafted clauses that meet requirements

### Legal Review Recommendation
- Whether Phaedra (legal counsel) review is required
- Specific items to discuss with legal team
- Risk assessment

Document: {text}'''

        # 3. Q&A Prompt
        self.qa_prompt_template = """Use the following pieces of the NDA document to answer the question at the end.
Focus on providing accurate information about confidentiality obligations, parties involved, terms, and legal provisions.

If you don't know the answer based on the NDA content, just say that the information is not specified in this NDA.

NDA Context:
{context}

Question: {question}

Answer:"""

    def _setup_intent_classifier(self):
        """Setup intent classification system"""
        self.intent_classifier = ChatPromptTemplate.from_messages([
            ("system", """You are an intent classifier for an NDA analysis chatbot.
            Classify the user's message into one of these categories:

            1. SUMMARY - User wants a basic summary of the NDA
            2. LEGAL_ANALYSIS - User wants detailed legal compliance analysis
            3. QUESTION - User is asking specific questions about the NDA content
            4. GENERAL - General conversation or greetings

            Examples:
            - "Summarize this NDA" → SUMMARY
            - "Give me a legal analysis" → LEGAL_ANALYSIS
            - "Analyze for compliance" → LEGAL_ANALYSIS
            - "Check this against our requirements" → LEGAL_ANALYSIS
            - "What are the confidentiality obligations?" → QUESTION
            - "Who are the parties?" → QUESTION
            - "Hello" → GENERAL

            Respond with only one word: SUMMARY, LEGAL_ANALYSIS, QUESTION, or GENERAL"""),
            ("user", "{user_message}")
        ])
        self.intent_chain = self.intent_classifier | self.llm | StrOutputParser()

    def load_nda_document(self, pdf_path: str) -> bool:
        """Load NDA PDF document"""
        try:
            print(f"📁 Loading NDA document: {pdf_path}")
            loader = PyPDFLoader(pdf_path)
            self.documents = loader.load()
            self.pdf_path = pdf_path
            print(f"✅ NDA loaded successfully! ({len(self.documents)} pages)")
            return True
        except Exception as e:
            print(f"❌ Error loading NDA: {str(e)}")
            return False

    def generate_document_summary(self) -> str:
        """Generate a basic document summary"""
        if not self.documents:
            return "❌ No NDA document loaded"

        try:
            print("📋 Generating document summary...")
            summarize_chain = load_summarize_chain(
                llm=self.llm,
                chain_type="stuff",
                prompt=ChatPromptTemplate.from_template(self.summary_prompt)
            )
            result = summarize_chain.invoke({"input_documents": self.documents})
            summary = result.get("output_text", str(result))
            print("✅ Summary generated!")
            return summary
        except Exception as e:
            return f"❌ Error generating summary: {str(e)}"

    def perform_legal_analysis(self) -> str:
        """Perform detailed legal compliance analysis using your requirements"""
        if not self.documents:
            return "❌ No NDA document loaded"

        try:
            print("⚖️ Performing legal compliance analysis...")
            analysis_chain = load_summarize_chain(
                llm=self.llm,
                chain_type="stuff",
                prompt=ChatPromptTemplate.from_template(self.legal_analysis_prompt)
            )
            result = analysis_chain.invoke({"input_documents": self.documents})
            analysis = result.get("output_text", str(result))
            print("✅ Legal analysis completed!")
            return analysis
        except Exception as e:
            return f"❌ Error performing legal analysis: {str(e)}"

    def setup_rag_chain(self):
        """Setup RAG chain for Q&A functionality"""
        if not self.documents:
            return None

        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""],
        )
        chunks = text_splitter.split_documents(self.documents)

        # Create vectorstore
        self.vectorstore = Chroma.from_documents(
            chunks,
            embedding=self.embeddings,
            persist_directory="./nda_chroma_db"
        )

        # Create QA prompt
        qa_prompt = PromptTemplate(
            template=self.qa_prompt_template,
            input_variables=["context", "question"]
        )

        # Create retrieval QA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(search_kwargs={"k": 4}),
            chain_type_kwargs={"prompt": qa_prompt},
            return_source_documents=True
        )
        return qa_chain

    def get_conversation_context(self, max_exchanges: int = 3) -> str:
        """Get recent conversation context for continuity"""
        history = self.get_conversation_history()
        if not history:
            return ""

        # Get last few exchanges
        recent = history[-max_exchanges:] if len(history) > max_exchanges else history
        context_parts = []

        for exchange in recent:
            context_parts.append(f"User: {exchange['user'][:100]}...")
            context_parts.append(f"Assistant: {exchange['assistant'][:100]}...")

        return "\n".join(context_parts)

    def ask_question(self, question: str) -> Dict[str, Any]:
        """Answer specific questions about the NDA using RAG with conversation context"""
        try:
            qa_chain = self.setup_rag_chain()
            if qa_chain is None:
                return {"answer": "❌ No NDA document loaded for Q&A"}

            # Add conversation context to the question
            conversation_context = self.get_conversation_context()
            if conversation_context:
                contextual_question = f"""Previous conversation context:
{conversation_context}

Current question: {question}

Please answer the current question while being aware of our previous discussion."""
            else:
                contextual_question = question

            result = qa_chain({"query": contextual_question})
            return {
                "answer": result["result"],
                "source_documents": result["source_documents"]
            }
        except Exception as e:
            return {"answer": f"❌ Error answering question: {str(e)}"}

    def classify_intent(self, user_message: str) -> str:
        """Classify user intent"""
        try:
            intent = self.intent_chain.invoke({"user_message": user_message}).strip().upper()
            return intent if intent in ["SUMMARY", "LEGAL_ANALYSIS", "QUESTION", "GENERAL"] else "QUESTION"
        except:
            return "QUESTION"

    def chat(self, user_message: str) -> Dict[str, Any]:
        """Main chat interface with enhanced conversation memory"""
        if not self.documents:
            return {
                "response": "❌ Please load an NDA document first using load_nda_document(pdf_path)",
                "intent": "ERROR",
                "sources": []
            }

        print(f"💬 User: {user_message}")

        # Classify intent
        intent = self.classify_intent(user_message)
        print(f"🎯 Intent: {intent}")

        # Get conversation context for continuity
        conversation_context = self.get_conversation_context()

        if intent == "SUMMARY":
            print("📄 Generating document summary...")
            if conversation_context:
                # Check if user is asking for summary of specific aspects based on previous conversation
                contextual_prompt = f"""Previous conversation context:
{conversation_context}

The user is now asking for a summary. Based on our previous discussion, provide a document summary that's relevant to our conversation flow.

{self.summary_prompt}"""
                summarize_chain = load_summarize_chain(
                    llm=self.llm,
                    chain_type="stuff",
                    prompt=ChatPromptTemplate.from_template(contextual_prompt)
                )
                result = summarize_chain.invoke({"input_documents": self.documents})
                response = result.get("output_text", str(result))
            else:
                response = self.generate_document_summary()
            sources = []

        elif intent == "LEGAL_ANALYSIS":
            print("⚖️ Performing legal compliance analysis...")
            if conversation_context:
                # Provide analysis with awareness of previous discussion
                contextual_analysis_prompt = f"""Previous conversation context:
{conversation_context}

Based on our previous discussion, please provide a legal analysis that addresses our conversation flow.

{self.legal_analysis_prompt}"""
                analysis_chain = load_summarize_chain(
                    llm=self.llm,
                    chain_type="stuff",
                    prompt=ChatPromptTemplate.from_template(contextual_analysis_prompt)
                )
                result = analysis_chain.invoke({"input_documents": self.documents})
                response = result.get("output_text", str(result))
            else:
                response = self.perform_legal_analysis()
            sources = []

        elif intent == "QUESTION":
            print("❓ Searching NDA for answer...")
            qa_result = self.ask_question(user_message)
            response = qa_result["answer"]
            sources = qa_result.get("source_documents", [])

        else:  # GENERAL
            print("💬 Handling general conversation...")
            conversation_context = self.get_conversation_context()

            general_prompt = f"""You are an NDA analysis assistant. You can help with:

1. **Document Summary** - Basic overview of the NDA
2. **Legal Analysis** - Detailed compliance check against firm requirements
3. **Q&A** - Answer specific questions about the document

Current conversation context: The user has an NDA document loaded and ready for analysis.

{f"Previous conversation context: {conversation_context}" if conversation_context else ""}

User message: {user_message}

Respond naturally and helpfully, taking into account our previous conversation:"""

            response = self.llm.invoke(general_prompt).content
            sources = []

        # Store in memory - ensure response is always a string
        response_str = self._ensure_string_response(response)
        self.memory.chat_memory.add_user_message(user_message)
        self.memory.chat_memory.add_ai_message(response_str)

        # Preview response
        preview = response_str[:200] + "..." if len(response_str) > 200 else response_str
        print(f"🤖 Assistant: {preview}")
        if sources:
            print(f"📚 Found {len(sources)} relevant document sections")

        return {
            "response": response_str,
            "intent": intent,
            "sources": sources
        }

    def get_conversation_history(self) -> List[Dict[str, str]]:
        """Get formatted conversation history"""
        history = []
        messages = self.memory.chat_memory.messages
        for i in range(0, len(messages), 2):
            if i + 1 < len(messages):
                human_msg = messages[i]
                ai_msg = messages[i + 1]
                history.append({
                    "user": human_msg.content,
                    "assistant": ai_msg.content
                })
        return history

    def clear_memory(self):
        """Clear conversation memory"""
        self.memory.clear()
        print("🗑️ Chat history cleared")

    def show_conversation_summary(self):
        """Display a summary of the current conversation"""
        history = self.get_conversation_history()
        if not history:
            print("💭 No conversation history yet")
            return

        print(f"💬 Conversation Summary ({len(history)} exchanges):")
        print("-" * 50)

        for i, exchange in enumerate(history[-5:], 1):  # Show last 5 exchanges
            print(f"{i}. User: {exchange['user'][:80]}...")
            print(f"   Bot: {exchange['assistant'][:80]}...")
            print()
