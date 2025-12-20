import QtQuick 
import QtQuick.Window 
import QtQuick.Controls 
import QtQuick3D 


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
    property real scaleFactor: 1.0
    property vector2d lastMousePos: Qt.vector2d(0, 0)
    property bool draggingWindow: false
    property bool rotatingModel: false
    property bool panningModel: false
    // 選取到的模型節點
    property Node selectedNode: null

    property string userInput: "" 

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
        spacing: 10

        environment: SceneEnvironment {
            backgroundMode: SceneEnvironment.Color
            clearColor: "transparent"
        }
        PerspectiveCamera {
            id: cam
            position: Qt.vector3d(0, 150, 350)
            eulerRotation.x: -15
        }
        DirectionalLight {
            eulerRotation: Qt.vector3d(-45, 0, 0)
            brightness: 1.8
        }
        // == == == == GLB 模型 == == == ==
        Model {
            id: ilulu
            source: "ilulu.glb"
            scale: Qt.vector3d(scaleFactor, scaleFactor, scaleFactor)
            position: Qt.vector3d(0, 0, 0)
            MouseArea {
                anchors.fill: parent
                onClicked: {
                    var node = view.pick(mouse.x, mouse.y)
                    if (node) {
                        console.log("Selected Node:", node.objectName)
                    }
                }
            }
        }
        // == == == == 加載 Rocking Chair 模型 == == == ==
        Model {
            id: rockingChair
            source: "uploads_files_3351752_Rocking_Chair2.obj"
            scale: Qt.vector3d(scaleFactor, scaleFactor, scaleFactor)
            position: Qt.vector3d(100, 0, 0) // 調整 Rocking Chair 的位置，避免與 ilulu 重疊
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
        font.family: "Microsoft JhengHei" // 設置字體
        font.pixelSize: 18 // 設置字體大小

        // 監聽文本變化
        onTextChanged: {
            // 當用戶輸入時更新 `userInput`
            userInput = inputBox.text
        }
        Keys.onPressed: {
            // 當按下回車鍵時，執行提交操作
            if([Qt.Key_Return,Qt.Key_Enter].includes(event.key)){
                event.accepted = true
                if(!(event.modifiers & (Qt.ShiftModifier | Qt.ControlModifier | Qt.AltModifier))){
                    animButton.clicked()
                    IC.input_line(userInput) // 執行失敗時同時不執行下一行
                    inputBox.text=""
                }
            }
        }
    }

    // 顯示用戶輸入的文本
    Text {
        id:message
        text: "輸入內容: " + userInput //  ***回報 和 回應
        anchors.top: inputBox.bottom
        anchors.left: inputBox.left
        color: "white"
        font.pixelSize: 16
    }
    // ***讀取到模型，卻看不見
    Button {
        id: animButton
        text: "輸出動畫"
        onClicked: {
            // 拆字判斷動畫
            real target = keyword_map[userInput];
            if (!target) {
                console.log("❌ 找不到 userInput", userInput);
                return;
            }
            real clipName =Object.keys(target)[0];
            // 搜尋 glTF 中的動畫列表
            for (real i = 0; i < ilulu.animations.length; i++) {
                real a = ilulu.animations[i];
                if (a.name === clipName) {
                    console.log("▶ 播放動畫:", clipName);
                    ilulu.animations[i].position = a.start;
                    ilulu.animations[i].duration = a.duration;
                    ilulu.animations[i].running = true;
                    return;
                }else if (a.name === "Idle"){
                    console.log("▶ 播放待機動畫:", clipName);
                    ilulu.animations[i].position = a.start;
                    ilulu.animations[i].duration = a.duration;
                    ilulu.animations[i].running = true;
                }
            }
        }
    }

    Button {
        id: quit
        text: "關閉"
        y: 10 + 2 * margin
        z: 99
        onClicked: {
            Qt.quit()
            IC.quitApp()
        }
    }
    
    // 新增與移除任務
    Column  {
        id: listErrand 
        spacing: 6
    }

    Rectangle  {
        id:listErrandButton
        text: "任務欄"
        anchors.bottom: parent.bottom
        anchors.horizontalCenter: parent.horizontalCenter
        z: 99

        TextField {
            id: ListNameField
            anchors.top: parent.top
            width: 240
            placeholderText: "輸入名稱"
        }

        TextField {
            id: nameField
            anchors.top: parent.top+25
            width: 240
            placeholderText: "輸入名稱"
        }

        MouseArea{
            anchors.fill: parent
            drag.target: listErrandButton
            property real dx =0
            property real dy =0
            acceptedButtons: Qt.LeftButton
            onPressed: {
                if (mouse.button === Qt.LeftButton) {
                    lastMousePos = Qt.vector2d(mouse.x, mouse.y)
                    // **重新命名
                    listErrandButton.text=ListNameField.text
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
                                            lastMousePos = Qt.vector2d(mouse.x, mouse.y)
                                        }
                                    }
                                    
                                    onReleased: {
                                        Item indexObj=listErrand.children[listErrand.children.indexOf(errandItem)]
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
                lastMousePos = Qt.vector2d(mouse.x, mouse.y);
                } 
            }
            else if (mouse.button === Qt.RightButton) { 
                rotatingModel = true;
                lastMousePos = Qt.vector2d(mouse.x, mouse.y);
            }
            else if (mouse.button === Qt.MiddleButton) {
            panningModel = true;
            lastMousePos = Qt.vector2d(mouse.x, mouse.y);
            }
        }

        onPositionChanged: {
            if (draggingWindow) {
                root.x += mouse.x - lastMousePos.x;
                root.y += mouse.y - lastMousePos.y;
            }
            if (rotatingModel) {
                ilulu.eulerRotation.y += mouse.x - lastMousePos.x;
                ilulu.eulerRotation.x += mouse.y - lastMousePos.y;
                lastMousePos = Qt.vector2d(mouse.x, mouse.y);
            }
            if (panningModel) {
                ilulu.position.x += (mouse.x - lastMousePos.x) * 0.5;
                ilulu.position.y -= (mouse.y - lastMousePos.y) * 0.5;
                lastMousePos = Qt.vector2d(mouse.x, mouse.y);
            }
        }

        onReleased: { draggingWindow = false; rotatingModel = false; panningModel = false }
    }

    MouseArea {
        anchors.fill: parent

        onPressed: {
            lastMousePos = Qt.vector2d(mouse.x, mouse.y);
            draggingWindow = true;
        }

        onPositionChanged: {
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

    WheelHandler {
        onWheel: root.scaleFactor += wheel.angleDelta.y * 0.001
    }

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
                text: "掛到 ilulu (root)"
                onClicked: {
                    if (selectedNode)
                        selectedNode.parent = ilulu;
                }
            }

            Button {
                text: "掛到 cam"
                onClicked: {
                    if (selectedNode)
                        selectedNode.parent = cam;
                }
            }
        }
    }
}
