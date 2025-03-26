```mermaid
flowchart TD
    subgraph Input_Processing ["Xử Lý Đầu Vào"]
        direction TB
        V["Video Bài Giảng"] -->|"Whisper (STT + Timestamps)"| TXT_A["Văn bản Audio + Timestamps"]
        S["File Slide (PDF/PPTX)"] -->|"Trích xuất Text/OCR"| TXT_S["Văn bản Slide + Slide Numbers"]
        V -->|"Vision (Keyframe + OCR)"| TXT_S_Vid["Văn bản Slide từ Video + Timestamps"]

        TXT_A -->|"Chunking & Embedding"| DB[("Vector Database\n+ Metadata: Timestamps, Slide#, Source")]
        TXT_S -->|"Chunking & Embedding"| DB
        TXT_S_Vid -->|"Chunking & Embedding"| DB
    end

    subgraph Agent_Tools ["Agent & Tool"]
        direction TB
        UM["User Message"] --> RA{"Router Agent"}
        RA -->|"RAG Query"| DB
        DB -->|"Retrieved Context"| RA

        RA -->|"Quyết định Tool"| T_RAG["/RAG Tool/"]
        RA -->|"Quyết định Tool"| T_Web["/Web Search Tool/"]
        RA -->|"Quyết định Tool"| T_Quiz["/Quiz Generator Tool/"]
        RA -->|"Quyết định Tool"| T_Sum["/Summary Tool/"]
        RA -->|"Quyết định Tool"| T_FC["/Flashcard Tool/"]
        RA -->|"Quyết định Tool"| T_Plan["/Study Plan Tool/"]
        RA -->|"Quyết định Tool"| T_Explain["/Concept Explainer Tool/"]
        
        T_RAG --> RA
        T_Web -->|"External Info"| RA
        T_Quiz --> RA
        T_Sum --> RA
        T_FC --> RA
        T_Plan --> RA
        T_Explain --> RA

        DB -->|"Analyze Content"| PA["Phân tích Chủ động"]
        PA -->|"Suggestions/Insights"| RA
    end

    subgraph Output_Generation ["OUTPUT "]
        direction TB
        RA -->|"Final Answer Generation"| LLM_Final["LLM - Generate Final Response"]
        LLM_Final --> OUT_Text["Output Text"]

        T_Sum -->|"Summary Text"| TTS["Text-to-Speech"]
        TTS --> VC["Voice Customizer"] --> OUT_Audio["Output Audio Summary"]

        RA -->|"Data for Viz"| VIZ["Tạo Trực quan hóa\nMindmap/Concept Links"]
        VIZ --> OUT_Viz["Output Hình ảnh/Interactive"]

        OUT_Text --> UserFeedback["User Feedback"]
        UserFeedback --> RA
    end
