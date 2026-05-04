import QtQuick 
import QtQuick3D 
import QtQuick.Window 
import QtQuick.Controls 
import Qt5Compat.GraphicalEffects // 必須導入此模組來使用 Glow


Window {
    id: root
    property int window_w: 480
    property int window_h: 360
    property int margin: 10
    property int text_h: 50
    property int button_w: 90
    property int button_h: 65

    width: window_w
    height: window_h
    color: "transparent"
    visible: true
    flags: Qt.FramelessWindowHint | Qt.Tool | Qt.WindowStaysOnTopHint
    property real scaleFactor: 50
    property vector2d lastMousePos: Qt.vector2d(0, 0)
    property bool draggingWindow: false
    property bool rotatingModel: false
    property bool panningModel: false
    // 選取到的模型節點
    property Node selectedNode: null

    property string userInput: "" 
    property real swY:30

    property var keyword_map: {
    "開心": {"happy": 1.0},
    "難過": {"sad": 1.0},
    "驚訝": {"surprise": 1.0},
    "眨眼": {"blink": 1.0},
    "張嘴": {"mouthOpen": 1.0}
    }

    View3D {
        id: view
        anchors.fill: parent
        //importScene: sceneRoot

        environment: SceneEnvironment {
            backgroundMode: SceneEnvironment.Color
            clearColor: "transparent"
        }
        PerspectiveCamera {
            id: cam
            position: Qt.vector3d(20, 158, 100)
            eulerRotation.x: -30
        }
        DirectionalLight {
            eulerRotation: Qt.vector3d(-45, 0, 0)
            brightness: 1.8
        }
        // == == == == GLB 模型 == == == ==
        Ilulu { 
            id: iluluModel
            scale: Qt.vector3d(scaleFactor*0.03, scaleFactor*0.03, scaleFactor*0.03)
            position: Qt.vector3d(0, 0, 0)
            
            SequentialAnimation on eulerRotation.y {
                loops: Animation.Infinite // 無限循環
                running: true             // 確保自動開始
                NumberAnimation {
                    from: -swY
                    to: swY+18
                    duration: 1230
                    easing.type: Easing.InOutQuad // 加入緩動效果，轉向時更平滑
                }
                NumberAnimation {
                    from: swY+18
                    to: -swY
                    duration: 1230
                    easing.type: Easing.InOutQuad
                }
            }
        }
    }

    TextArea {
        id: inputBox
        width: window_w*0.8
        height: Math.max(50, Math.min(contentHeight, window_h * 0.4))
        x: (parent.width - width) / 2  // 水平居中
        y: (parent.height - height)-margin-text_h // 垂直居下
        text:""
        wrapMode: Text.Wrap // 自動換行
        placeholderText: "請輸入windowTittle, path, action... (:多重路徑、::分行、<>錄製)"
        focus: true // 點擊即可輸入
        // font.family: "Microsoft JhengHei" // 設置字體
        font.pixelSize: 18 // 設置字體大小
        color: Qt.rgba(0.1, 0.0, 0.15, 0.9)
        font.family: "Courier" // 用等寬字體更有駭客感


        background: Rectangle {
            color: Qt.rgba(0.68, 1, 0.18, 0.4) 
            radius: 8
            border.color: Qt.rgba(0.68, 1, 0.18, 1)  // 深一點的邊框讓邊界更清晰
            border.width: 1
            }
        // 監聽文本變化
        onTextChanged: {
            if(text.length > 0) {  // *** 進入 計算物體實際大小的 抓取模式
                if (!/^(.*)_W(\d+)_H(\d+)_Z([\d.]+)\.png$/.test(text)) {}
            // 當用戶輸入時更新 `userInput`
            userInput = text
            }
        }
        Keys.onPressed: (event) => {
            // 當按下回車鍵時，執行提交操作
            if([Qt.Key_Return,Qt.Key_Enter].includes(event.key)){
                event.accepted = true
                if(!(event.modifiers & (Qt.ShiftModifier | Qt.ControlModifier | Qt.AltModifier))){
                    animButton.clicked()
                    IC.input_line(userInput) // 執行失敗時同時不執行下一行
                    text=""
                }   
            }
        }
    }

    Glow {
        id: textGlow
        anchors.fill: hackerText
        source: hackerText
        radius: 8
        samples: 17
        color: "#BC13FE" // 電光紫，與你的深紫文字形成層次感
        spread: 0.2

        // 閃爍動畫 (呼吸燈效果)
        SequentialAnimation on opacity {
            loops: Animation.Infinite
            NumberAnimation { from: 0.4; to: 1.0; duration: 1500; easing.type: Easing.InOutQuad }
            NumberAnimation { from: 1.0; to: 0.4; duration: 1500; easing.type: Easing.InOutQuad }
        }
    }
    
    // 顯示用戶輸入的文本
    Item {
        anchors.top: inputBox.bottom
        anchors.left: inputBox.left
        anchors.right: parent.right // 建議加上 right，換行 (wrap) 才會生效
        // 底色文字 (深紫色)
        Text {
            id: baseText
            text: "引頸期盼地等待訊息..." //  ***回報 和 回應
            color: Qt.rgba(0.435, 0.306, 0.216, 1.0)
            font.family: "Courier" // 用等寬字體更有駭客感
            font.pixelSize: 16
            wrapMode: Text.Wrap // 讓長對話自動換行
            
            Connections {
                target: Backend // 確保這裡對應到你 Python 注入的名稱
                function onResponseUpdated(all_text) {
                    message.text= "回應: " + all_text
                }
            }
            Connections {
                target: neonFlow
                function onPosChanged() {
                    // 當掃描線走到中間 (例如 0.3 到 0.7 之間) 時，隨機跳動
                    if (neonFlow.pos > 0.3 && neonFlow.pos < 0.7) {
                        baseText.x = baseText.x + (Math.random() * 3 - 1); 
                    } else {
                        baseText.x = parent.width / 2 - baseText.width / 2; // 回歸正中
                    }
                }
            }
        }
        // 霓虹斜線漸層
        // 用來製作掃描光澤的文字層
        Text {
            id: maskText
            text: baseText.text
            font: baseText.font
            anchors.fill: baseText
            visible: false // 隱藏起來，只作為遮罩使用
        }

        // 霓虹斜線漸層
        LinearGradient {
            id: neonFlow
            anchors.fill: baseText
            source: baseText // 直接把文字當來源
            // 斜率：調整 point(x, y) 可以改變斜線的角度
            start: Qt.point(0, 0)
            end: Qt.point(baseText.width * 0.5, baseText.height) 

            property real pos: -1.0

            gradient: Gradient {
                // 第 1 條細線
                GradientStop { position: neonFlow.pos; color: "transparent" }
                GradientStop { position: neonFlow.pos + 0.05; color: Qt.rgba(1, 0, 1, 0.3) } 
                GradientStop { position: neonFlow.pos + 0.1; color: "#FF00FF" } // 粉紅細線
                GradientStop { position: neonFlow.pos + 0.05; color: Qt.rgba(1, 0, 1, 0.3) } 
                GradientStop { position: neonFlow.pos + 0.2; color: "transparent" }
                
                // 間隔
                GradientStop { position: neonFlow.pos + 0.3; color: "transparent" }
                
                // 第 2 條細線 (稍微寬一點點，增加層次)
                GradientStop { position: neonFlow.pos + 0.38; color: Qt.rgba(0.03, 0.58, 0.97, 0.3) }
                GradientStop { position: neonFlow.pos + 0.4; color: "#0994f8" } // 青色細線
                GradientStop { position: neonFlow.pos + 0.38; color: Qt.rgba(0.03, 0.58, 0.97, 0.3) }
                GradientStop { position: neonFlow.pos + 0.6; color: "transparent" }
                
                // 間隔
                GradientStop { position: neonFlow.pos + 0.55; color: "transparent" }
                
                // 第 3 條細線
                 GradientStop { position: neonFlow.pos + 0.82; color: Qt.rgba(0.62, 0.24, 0.93, 0.3) }
                GradientStop { position: neonFlow.pos + 0.85; color: "#9f3eee" }
                 GradientStop { position: neonFlow.pos + 0.82; color: Qt.rgba(0.62, 0.24, 0.93, 0.3) }
                GradientStop { position: neonFlow.pos + 0.92; color: "transparent" }
            }

            PropertyAnimation on pos {
                from: -2
                to: 3.6
                duration: 3500
                loops: Animation.Infinite
                easing.type: Easing.Linear // 線性移動更像機器掃描
            }
        }
        // 加一點點整體外發光，讓霓虹感更重
        Glow {
            anchors.fill: neonFlow
            source: neonFlow
            radius: 4
            samples: 9
            color: "#00FFFF"
            spread: 0.2
        }
    }
    // 顯示回覆用戶的文本
        // TODO:收到的路徑轉成句子，
        // TODO:*** 打開路徑下的所有圖片
        // TODO:*** 點擊某個詞，顯示該詞對應路徑下的圖片集
        // TODO:*** 動態圖片：每次撥放幾張圖，循環撥放
        // pyside6-balsam *.glb 另存成 iluluModel.qml and meshes/*.mesh and maps/模型使用的貼圖 
    Column {
        spacing: 10
        Text {
            id: dialogue
            text: Backend ? Backend.path_dir : "" 
            color: "white"
            font.pixelSize: 16

            MouseArea {
                anchors.fill: parent
                onClicked: (mouse) => { // 點擊文字請求圖片列表
                imgTimer.stop()       // 切換前先停止播放
                Backend.getImages(dialogue.text) // 傳文字給後端
                }  
            }
        }

        Image {
            id: imgDisplay
            width: 300
            height: 300
            fillMode: Image.PreserveAspectFit
        }

        Timer {
            id: imgTimer
            interval: 1000 // 每秒切換
            repeat: true
            running: false
            onTriggered: {
                if (imageList.length > 0) {
                    imgDisplay.source = imageList[currentIndex]
                    currentIndex = (currentIndex + 1) % imageList.length
                } else {
                imgDisplay.source = "" // 空列表清除圖片
            }
            }
        }

        property var imageList: []
        property int currentIndex: 0

        Connections {
            target: Backend
            function onImagesReady(images) {
                imageList = images
                currentIndex = 0
                if (imageList.length > 0) {
                    imgDisplay.source = imageList[0] // 立即顯示第一張
                    imgTimer.start()
                } else {
                    imgDisplay.source = ""
                    imgTimer.stop()
                }
            }
        }
    }


    Button {
        id: animButton
        text: "輸出動畫"
        opacity: listErrandButton.x < 50 ? 0 : 1
        Behavior on opacity { NumberAnimation { duration: 300 } }

        onClicked: (mouse) => {
            // 拆字判斷動畫
            var target = keyword_map[userInput];
            if (!target) {
                console.log("動畫❌ 找不到 userInput", userInput);
                return;
            }
            var clipName  =Object.keys(target)[0];
            // 搜尋 glTF 中的動畫列表
            for (var i = 0; i < iluluModel.animations.length; i++) {
                var a = iluluModel.animations[i];
                if (a.name === clipName) {
                    console.log("▶ 播放動畫:", clipName);
                    iluluModel.animations[i].position = a.start;
                    iluluModel.animations[i].duration = a.duration;
                    iluluModel.animations[i].running = true;
                    return;
                }else if (a.name === "Idle"){
                    console.log("▶ 播放待機動畫:", clipName);
                    iluluModel.animations[i].position = a.start;
                    iluluModel.animations[i].duration = a.duration;
                    iluluModel.animations[i].running = true;
                }
            }
        }
    }

    Button {
        id: quit
        text: "關閉"
        y: 10 + 2 * margin
        z: 99
        opacity: listErrandButton.x < 50 ? 0 : 1
        Behavior on opacity { NumberAnimation { duration: 300 } }

        onClicked: (mouse) => {
            Qt.quit()
        }
    }
    
    // 新增與移除任務
    Column  {
        id: listErrand 
        spacing: 6
    }

    Rectangle  {
        id:listErrandButton
        anchors.bottom: parent.bottom
        anchors.horizontalCenter: parent.horizontalCenter
        z: 99
        property bool dragging: false

        Text {
            id:title
            text:"任務欄"
            anchors.centerIn: parent
            color:"white"
        }


        TextField {
            id: listNameField
            anchors.top: parent.top
            width: 240
            placeholderText: "輸入名稱"
        }

        TextField {
            id: nameField
            anchors.top: parent.top 
            anchors.topMargin: 25  // 偏移量要分開寫
            width: 240
            placeholderText: "輸入名稱"
        }

        MouseArea{
            anchors.fill: parent
            drag.target: listErrandButton
            property real dx :0
            property real dy :0
            acceptedButtons: Qt.LeftButton
            onPressed: {
                if (mouse.button === Qt.LeftButton) {
                    lastMousePos = Qt.point(mouse.x, mouse.y)
                    // **重新命名
                    title.text = listNameField.text
                }
            }

            onPositionChanged: {
                // 按鈕位移後，縮小或放大 整個任務欄
                dx = mouse.x - lastMousePos.x
                dy = mouse.y - lastMousePos.y
                if ( Math.abs(dx)>20 ||  Math.abs(dy)>20 ) {
                    if(listErrandButton.height<button_h){
                        // ***任務欄 全部顯示
                        listErrandButton.height=button_h
                    }else{
                        // ***只顯示 按鈕
                        listErrandButton.height=50
                    }
                }
                else{
                    listErrandButton.dragging=false
                }
            }
            onReleased: {
                // **放大時，按鈕不位移後，增加任務
                if(!listErrandButton.dragging && listErrandButton.height>=button_h){
                    
                    Qt.createQmlObject('
                        Item {
                            id: errandItem
                            width: parent.width
                            height: parent.height-2.5

                            Rectangle {
                                anchors.fill: parent
                                color: "#333"
                                radius: 6
                                
                                TextArea { 
                                    property bool posD: false
                                    id: errand
                                    width: 300
                                    height: 80
                                    text: nameField.text
                                    anchors.fill: parent
                                    anchors.margins: 6
                                    wrapMode: TextArea.Wrap
                                }
                                MouseArea {
                                    anchors.fill: parent
                                    drag.target: errand
                                    acceptedButtons: Qt.LeftButton
                                    onPressed: {
                                        if (mouse.button === Qt.LeftButton) {
                                            lastMousePos = Qt.point(mouse.x, mouse.y)
                                        }
                                    }
                                    
                                    onReleased: {
                                        var indexObj=listErrand.children[listErrand.children.indexOf(errandItem)]
                                        bool objOk=false
                                        if ( Math.abs(mouse.x - lastMousePos.x)>20 && indexObj!==-1 ) {
                                            // **任務左右位移時移除 ，GPT 寫錯 GPT已死
                                            indexObj.destroy(); 
                                        }else if ( Math.abs( mouse.y - lastMousePos.y)>20 ) {
                                            // ***任務上下位移時變更順序，放開在哪一個子物件上面，該順序以後的全部子物件都後移一位
                                            for (real child of listErrand.children) {
                                                if(listErrand.children.indexOf(child)==1)
                                                    objOk=false
                                                if (mouse.y >= child.y && mouse.y < child.y + child.height){
                                                    child.parent = null;
                                                    child.parent = listErrand;
                                                    listErrand.stackBefore(child);
                                                    objOk=true
                                                }
                                                if(objOk){
                                                    child.parent = null;
                                                    child.parent = listErrand;
                                                }
                                            }
                                        }else{
                                            // **無位移時重新命名
                                            errandItem.errand.text= nameField.text 
                                            nameField.text=""
                                        }
                                    }
                                }
                            }
                        }
                        ', listErrand
                    )
                    nameField.text = ""
                }
            }
        }
    }
    // end 新增與移除任務


    // == = 視窗拖曳 == =
    MouseArea {
        anchors.fill: parent
        acceptedButtons: Qt.LeftButton

        onPressed: {
            if (mouse.button === Qt.LeftButton) {
                if (!rotatingModel && !panningModel) {
                draggingWindow = true;
                lastMousePos = Qt.point(mouse.x, mouse.y);
                } 
            }
            else if (mouse.button === Qt.RightButton) { 
                rotatingModel = true;
                lastMousePos = Qt.point(mouse.x, mouse.y);
            }
            else if (mouse.button === Qt.MiddleButton) {
            panningModel = true;
            lastMousePos = Qt.point(mouse.x, mouse.y);
            }
        }

        onPositionChanged: {
            if (draggingWindow) {
                root.x += mouse.x - lastMousePos.x;
                root.y += mouse.y - lastMousePos.y;
            }
            if (rotatingModel) {
                iluluModel.eulerRotation.y += mouse.x - lastMousePos.x;
                iluluModel.eulerRotation.x += mouse.y - lastMousePos.y;
                lastMousePos = Qt.point(mouse.x, mouse.y);
            }
            if (panningModel) {
                iluluModel.position.x += (mouse.x - lastMousePos.x) * 0.5;
                iluluModel.position.y -= (mouse.y - lastMousePos.y) * 0.5;
                lastMousePos = Qt.point(mouse.x, mouse.y);
            }
        }

        onReleased: { draggingWindow = false; rotatingModel = false; panningModel = false }
    }

    MouseArea {
        anchors.fill: parent

        onPressed:(mouse)=> {
            lastMousePos = Qt.point(mouse.x, mouse.y);
            draggingWindow = true;
        }

        onPositionChanged:(mouse)=> {
            if (draggingWindow) {
                root.x += mouse.x - lastMousePos.x;
                root.y += mouse.y - lastMousePos.y;
                // 左右邊緣自動最小化
                if (root.x <= 0) {
                    root.width = 50; // 最小化寬度
                    root.x = 0; 
                    root.height = Math.max(50, window_h*(root.width/window_w)); 
                } else if (root.x+(window_w/2) >= Screen.width) {
                    root.width = 50;
                    root.x = Screen.width-50; 
                    root.height = Math.max(50, window_h*(root.width/window_w)); 
                } else {
                    root.width = window_w;
                    root.height = window_h;
                }
            }
        }

        onReleased: draggingWindow = false
    }


    // == = 滾輪縮放 == =

//    WheelHandler {
//        onWheel: root.scaleFactor += wheel.angleDelta.y * 0.001
//    }

    // == == == == UI：重新掛載子物件 == == == ==

    Rectangle {
        anchors.right: parent.right
        anchors.top: parent.top
        width: 200
        height: 200
        color: "#242424AA"
        radius: 8
        Column {
            anchors.fill: parent
            anchors.margins: 10
            spacing: 8
            Text {
                text: selectedNode ? "選取: " + selectedNode.objectName : "未選取節點"
                color: "white"
            }
            Button {
                text: "掛到 iluluModel (root)"
                onClicked: (mouse) => {
                    if (selectedNode)
                        selectedNode.parent = iluluModel;
                }
            }

            Button {
                text: "掛到 cam"
                onClicked: (mouse) => {
                    if (selectedNode)
                        selectedNode.parent = cam;
                }
            }
        }
    }
}
