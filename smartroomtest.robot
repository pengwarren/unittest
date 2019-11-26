* Settings *
Variables       smartroomtest.py

* Test Cases *
TestPerformNaiveBayes1
    ${luna.text}        Set Variable     please turn on the lights
    log to console  ${luna.polarities}
TestPerformNaiveBayes2
    ${luna.text}        Set Variable     please turn off the lights
    log to console  ${luna.polarities}
TestPerformNaiveBayes3
    ${luna.text}        Set Variable     please don't turn on the lights
    log to console  ${luna.polarities}
TestPerformNaiveBayes4
    ${luna.text}        Set Variable     please do not turn on the lights
    log to console  ${luna.polarities}
TestPerformNaiveBayes5
    ${luna.text}        Set Variable     please turn on and off the lights
    log to console  ${luna.polarities}
TestBuildTrainingData1
    ${luna.text}    set Variable    please turn off the lights
TestBuildTrainingData2
    ${luna.text}    set Variable    please turn off the lights
TestConvertSpeechToText1
    ${luna.text}    set Variable    please don't turn off the lights
TestConvertSpeechToText2
    ${luna.text}    set Variable    please don't turn off the lights
TestGetmicrophoneindex1
    ${luna.text}    set Variable    please do not turn on the lights
TestGetmicrophoneindex2
    ${luna.text}    set Variable    please do not turn on the lights
TestPerformClassification1
    ${luna.text}    set Variable    please do not turn on the lights
TestPerformClassification2
    ${luna.text}    set Variable    please do not turn on the lights
TestPerformRequest1
    ${luna.text}    set Variable    please do not turn on the lights
TestPerformRequest2
    ${luna.text}    set Variable    please do not turn on the lights
Testwaitforwakeword1
    ${luna.text}    set Variable    please do not turn on the lights
    ${luna.wait_for_wake_word}  set Variable    luna
    Should Be Equal ${luna.wait_for_wake_word}  ${luna.text}
Testwaitforwakeword2
    ${luna.text}    set Variable    luna
    ${luna.wait_for_wake_word}  set Variable    luna
    Should Be Equal ${luna.wait_for_wake_word}  ${luna.text}