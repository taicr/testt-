```mermaid
graph TD
    Start --> InitializeSystem

    subgraph SystemSetup
        InitializeSystem --> LoadDataAndModels
        LoadDataAndModels --> BuildVectorStore
    end

    BuildVectorStore --> ChatbotReady

    ChatbotReady --> WaitForUserInput
    WaitForUserInput -- UserSendsQuestion --> ProcessQuestion
    WaitForUserInput -- UserSendsExit --> EndChat

    subgraph QuestionHandling
        ProcessQuestion --> SearchRelevantVideos
        SearchRelevantVideos -- NoVideosFound --> PrepareNotFoundMessage
        SearchRelevantVideos -- VideosFound --> PrepareContextForLLM
        PrepareContextForLLM --> GetAnswerFromLLM
        GetAnswerFromLLM --> FormatOutput
    end

    PrepareNotFoundMessage --> DisplayResponseToUser
    FormatOutput --> DisplayResponseToUser
    DisplayResponseToUser --> WaitForUserInput

    EndChat --> End
