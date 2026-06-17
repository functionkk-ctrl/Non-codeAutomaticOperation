import QtQuick
import QtQuick3D
import QtQuick.Window
import QtQuick.Controls
import QtQuick.Effects
import Qt5Compat.GraphicalEffects // 必須導入此模組來使用 Glow // Qt 6 請使用此模組；Qt 5 請改為 import QtGraphicalEffects 1.15
import QtQuick.Layouts

Window {

    // 1. 中央 任務欄（新增與移除任務） //點擊任務欄內的空處即新增按鈕(CamButton)，按下按鈕即重新命名，左右拉即移除，上下拉即重新調整任務順位 //drag時的高度
    id: root
    property int window_w: 480
    property int window_h: 360
    property int margin: 10
    property int text_h: 50
    property int button_w: 80
    property int button_h: 55

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
    property real swY: 30

    property var keyword_map: {
        "開心": {
            "happy": 1.0
        },
        "難過": {
            "sad": 1.0
        },
        "驚訝": {
            "surprise": 1.0
        },
        "眨眼": {
            "blink": 1.0
        },
        "張嘴": {
            "mouthOpen": 1.0
        }
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
            position: Qt.vector3d(37, 158, 100)
            eulerRotation.x: -30
        }
        DirectionalLight {
            eulerRotation: Qt.vector3d(-45, 0, 0)
            brightness: 1.8
        }
        // == == == == GLB 模型 == == == ==
        Ilulu {
            id: iluluModel
            property real size: 0.88 // 1 幾乎占滿視窗整個高度
            scale: Qt.vector3d(scaleFactor * size, scaleFactor * size, scaleFactor * size)
            position: Qt.vector3d(0, 0, 0)

            SequentialAnimation on eulerRotation.y {
                loops: Animation.Infinite // 無限循環
                running: true             // 確保自動開始
                NumberAnimation {
                    from: -swY
                    to: swY + 18
                    duration: 1230
                    easing.type: Easing.InOutQuad // 加入緩動效果，轉向時更平滑
                }
                NumberAnimation {
                    from: swY + 18
                    to: -swY
                    duration: 1230
                    easing.type: Easing.InOutQuad
                }
            }
        }
    }

    // 1. 在這裡定義你的自訂按鈕組件 (Component)
    component CamButton: Button {
        id: control
        checkable: true
        implicitWidth: button_w
        implicitHeight: button_h
        //Layout.preferredWidth: button_w
        //Layout.preferredHeight: button_h

        // 宣告一個屬性別名，指向內部 mainText 的 text 屬性
        property alias buttonText: mainText.text
        property real maxFontSize: 28
        property int index: -1
        z: 9

        opacity: root.x < 50 ? 0 : 1
        Behavior on opacity {
            NumberAnimation {
                duration: 300
            }
        }

        onClicked: {
            let 最後一行剩幾個空白號才填滿 = (containerItem.width - containerItem.flowLayout.x) / containerItem.customSize; // 自動換行
            if (control.checked) {
                Backend.conversation("按下了" + mainText.text + "  ".repeat(最後一行剩幾個空白號才填滿));
            } else {
                Backend.conversation("取消了" + mainText.text + "  ".repeat(最後一行剩幾個空白號才填滿));
            }
        }
        // 純平背景，不需要任何外框陰影
        background: Rectangle {
            id: diamondRect
            color: Qt.rgba(156 / 255, 203 / 255, 244 / 255, 0.8) // "#9ccbf4"
            radius: 10
            // TODO:***鑽石外觀
        }
        // 主要文字內容
        contentItem: Item {
            id: textContainer
            //implicitWidth: mainText.implicitWidth
            //implicitHeight: mainText.implicitHeight
            //width: control.availableWidth
            //height: control.availableHeight
            //anchors.fill: parent
            //anchors.centerIn: parent // 讓文字與陰影居中在按鈕中央
            //Layout.alignment: Qt.AlignHCenter // Qt.AlignHCenter Qt.AlignLeft
            //enabled: false // ⭕ 修正 2：強制讓整個內容物不響應任何滑鼠點擊，點擊事件會直接穿透到 Button

            Text {
                id: mainText
                text: ""
                color: "white" // 文字主顏色
                anchors.centerIn: parent
                padding: 1

                // 關鍵設定 1：設定字型錨定與最大大小
                font.pixelSize: control.maxFontSize
                horizontalAlignment: Text.AlignHCenter
                verticalAlignment: Text.AlignVCenter
                // 關鍵設定 2：啟動自動縮小機制
                fontSizeMode: Text.Fit  // 當文字超出寬度時，自動縮小字型大小
                minimumPixelSize: 11     // 允許縮小到的最小極限（避免縮到 0 變看不見）
                // 關鍵設定 3：必須限制文字元件的寬度，否則它不知道何時該縮小
                width: control.availableWidth
                // 4. 精髓：如果字多到連 9 號字都塞不下，自動在末端加上 "..."
                elide: Text.ElideRight
                wrapMode: Text.NoWrap   // 不換行，強迫在單行內縮小
                // 當我們在外部用 MultiEffect 渲染時，這裡保持純文字即可
                //horizontalAlignment: Text.AlignHCenter
                //verticalAlignment: Text.AlignVCenter
                //visible: false // 因為我們只要 MultiEffect 渲染後的結果，這樣才不會有雙重重影 Bug
            }

            MultiEffect {
                anchors.fill: mainText
                source: mainText

                // 這樣一來，QML 就會隱藏原本的純白文字，只畫出經由 MultiEffect 加工後、帶有立體陰影的文字，徹底解決雙重文字重影 Bug！
                // 【核心邏輯 1：陰影要大、羽化陰影邊緣】
                shadowEnabled: true
                blurMax: 3             // 模糊最大半徑
                shadowBlur: 0.3        // 設為最大值 1.0 達到最強羽化效果
                shadowColor: "#60000052" // 半透明黑色

                // 【核心邏輯 2：未選取時外凸 (Raised)，已選取時內凹 (Sunken)】
                shadowHorizontalOffset: control.checked ? -2 : 2
                shadowVerticalOffset: control.checked ? -2 : 2

                autoPaddingEnabled: true  // 自動計算陰影與模糊所需的邊框留白
            }
        }
    }

    // == = 視窗拖曳 == = // MouseArea 必須在 GridLayout 之前
    MouseArea {
        anchors.fill: parent
        acceptedButtons: Qt.LeftButton

        onPressed: {
            if (mouse.button === Qt.LeftButton) {
                if (!rotatingModel && !panningModel) {
                    draggingWindow = true;
                    lastMousePos = Qt.point(mouse.x, mouse.y);
                }
            } else if (mouse.button === Qt.RightButton) {
                rotatingModel = true;
                lastMousePos = Qt.point(mouse.x, mouse.y);
            } else if (mouse.button === Qt.MiddleButton) {
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
        onReleased: {
            draggingWindow = false;
            rotatingModel = false;
            panningModel = false;
        }
    }
    // 視窗
    MouseArea {
        anchors.fill: parent

        onPressed: mouse => {
            lastMousePos = Qt.point(mouse.x, mouse.y);
            draggingWindow = true;
        }

        onPositionChanged: mouse => {
            if (draggingWindow) {
                root.x += mouse.x - lastMousePos.x;
                root.y += mouse.y - lastMousePos.y;
                // 左右邊緣自動最小化
                if (root.x <= 0) {
                    root.width = 50; // 最小化寬度
                    root.x = 0;
                    root.height = Math.max(50, window_h * (root.width / window_w));
                } else if (root.x + (window_w / 2) >= Screen.width) {
                    root.width = 50;
                    root.x = Screen.width - 50;
                    root.height = Math.max(50, window_h * (root.width / window_w));
                } else {
                    root.width = window_w;
                    root.height = window_h;
                }
            }
        }

        onReleased: draggingWindow = false
    }

    // CamButton
    // TaskBarItem
    // ContainerItem
    // InputBoxTextArea
    // ColButton
    // ImgItem
    // RectAinmate
    //0, 01234
    //1, 01234
    //2, 01234
    //3, 01234
    //4, 01234
    //ColButton 0,0
    //TaskBarItem 01234,4
    //RectAinmate 234,0
    //ContainerItem 234,123
    //ImgItem 23,4
    //InputBoxTextArea 4,0123
    GridLayout {
        //Layout.fillWidth: true;
        //Layout.fillHeight: true
        anchors.fill: parent
        columns: 5 // 修正：索引要到 4，總欄數必須是 5 (0,1,2,3,4)
        rows: 5    // 總列數為 5 (0,1,2,3,4)
        columnSpacing: 10
        rowSpacing: 6

        // 【ColButton】 座標 (0, 0) -> 第一列、第一欄
        ColButton {
            id: btn
            Layout.row: 0
            Layout.column: 0
            Layout.columnSpan: 2
            Layout.fillWidth: true
            Layout.fillHeight: true
            //Layout.preferredWidth: parent.width * 0.2 // 固定側邊欄比例
        }

        // 【TaskBarItem】 座標 (0, 4) 且跨 4 列 -> 右側整條側邊欄
        TaskBarItem {
            id: taskBar
            Layout.row: 0
            Layout.column: 4
            Layout.rowSpan: 5 // 縱跨 5 列 (0,1,2,3,4)
            Layout.fillWidth: true
            Layout.fillHeight: true
            Layout.preferredWidth: button_w * 2
            //implicitHeight:button_h
            //Layout.preferredHeight: Math.max(button_h, listErrand.height) // 高度完全由按鈕數量決定
            //Layout.alignment: Qt.AlignHCenter // Qt.AlignHCenter Qt.AlignLeft
        }

        // 【RectAinmate】 座標 (1, 0) 且跨 2 列 -> 左側中間動畫區
        RectAinmate {
            Layout.row: 1
            Layout.column: 0
            Layout.rowSpan: 2 // 縱跨 2 列 (1,2) // 0,3 放動畫角色
        }

        // 【containerItem】 座標 (1, 1) 跨 2 列 3 欄 -> 中央核心容器
        ContainerItem {
            id: containerItem
            Layout.row: 1
            Layout.column: 1
            Layout.rowSpan: 2    // 縱跨 2 列 (1,2)
            Layout.columnSpan: 3 // 橫跨 3 欄 (1,2,3)
            Layout.fillWidth: true
            Layout.fillHeight: true
            //implicitHeight: childrenRect.height
        }

        // 【InputBoxTextArea】 座標 (4, 0) 橫跨 4 欄 -> 底部輸入框
        InputBoxTextArea {
            id: inputTextArea
            Layout.fillWidth: true
            Layout.fillHeight: true
            Layout.row: 4
            Layout.column: 0
            Layout.columnSpan: 4 // 橫跨 4 欄 (0,1,2,3)
        }
    }
    component TaskBarItem: Item {
        Layout.fillWidth: true
        Layout.fillHeight: true
        width: parent ? parent.width : implicitWidth
        height: parent ? parent.height : implicitHeight
        property alias taskListModel: taskListModel

        //新增樣式
        ListModel {
            id: taskListModel

            ListElement {
                taskName: "初始按鈕"
                checked: false
                index: 0
            }
            // path_all(users/button) // TODO:*******讀取用戶按紐設定，創建後要有基本功能

        }

        Connections {
            target: Backend
            function onUsersUpdated(users) {
                if (users && users.length > 0) {
                    taskListModel.clear();
                    for (var i = 0; i < users.length; i++) {
                        taskListModel.append(users[i]);
                    }
                }
            }
        }

        Component.onCompleted: {
            Backend.getUsers();
        }

        //顯示範圍
        Rectangle {
            width: parent.width
            anchors.fill: parent
            color: Qt.rgba(0.435, 0.306, 0.216, 1.0)
            radius: 10

            Text {
                id: headerText
                anchors.top: parent.bottomY
                text: "任務欄"
                color: Qt.rgba(0.0, 1.0, 0.85, 1.0)
                font.pixelSize: 16
            }
        }
        //新增按鈕
        MouseArea {
            anchors.fill: parent
            //acceptedButtons: Qt.LeftButton
            propagateComposedEvents: true
            z: 0

            onClicked: mouse => {
                // 💡 既然能走到這一層，代表內部的按鈕都沒有被點擊（點到空白處）
                // 直接觸發新增任務！
                let 最後一行剩幾個空白號才填滿 = (containerItem.width - containerItem.flowLayout.x) / containerItem.customSize; // 自動換行
                Backend.conversation("點擊到空白處" + "  ".repeat(最後一行剩幾個空白號才填滿));
                createNewTaskButton();
            }
        }
        // 垂直排列任務按鈕
        GridLayout {
            columns: 1 // 上下拖曳，設定為單欄
            rowSpacing: 3
            Repeater {
                id: taskRepeater
                model: taskListModel

                delegate: CamButton {
                    id: currentButton
                    buttonText: model.taskName
                    checked: model.checked
                    index: model.index
                    Drag.active: dragArea.drag.active
                    Drag.source: currentButton
                    Drag.hotSpot.x: width / 2
                    Drag.hotSpot.y: height / 2

                    MouseArea {
                        id: dragArea
                        anchors.fill: parent
                        acceptedButtons: Qt.LeftButton
                        propagateComposedEvents: true // 垃圾行:讓已存在的按鈕點擊事件可以穿透

                        drag.target: currentButton
                        //drag.axis: Drag.XAndYAxis // 允許左右與上下拖動
                        property real startY: 0

                        onClicked: mouse => {
                            currentButton.clicked(); // 點擊事件傳給外層的 CamButton
                            handleButtonInteraction(currentButton, currentButton.index);
                            // ⚠️ 垃圾行:阻止點擊事件傳給底層的空白處，防止一邊點按鈕一邊新增任務
                            mouse.accepted = true;
                        }
                        onPressed: mouse => {
                            taskRepeater.Layout.fillWidth = false;
                            startY = currentButton.y;
                        }
                        onReleased: mouse => {
                            // TODO:其實按鈕的clicked無實際功能，但還是出錯了，只能是其他地方導致。 ***插隊異常延遲 ***重新命名異常觸發 ***新增按鈕異常觸發 ***重新排序後異常初始化 ***按下異常觸發替換順序
                            taskRepeater.Layout.fillWidth = true;
                            var startIndex = Math.floor((startY + currentButton.height / 2) / (button_h + 3)); // spacing 3
                            var toIndex = Math.floor((currentButton.y + currentButton.height / 2) / (button_h + 3)); // spacing 3
                            if (Math.abs(currentButton.x) > 30) {
                                taskListModel.remove(startIndex);
                            } else if (toIndex !== currentButton.index) {
                                taskListModel.move(currentButton.index, toIndex, 1);
                                currentButton.Drag.drop();
                            } else {
                                yAnimation.start();
                            }
                            currentButton.x = 0;
                            currentButton.index = toIndex;
                        }
                    }
                }
            }
        }

        // 💡 Function A: 新增按鈕
        function createNewTaskButton() {
            var txt = inputTextArea.text.trim();
            if (txt === "")
                txt = "新自動化任務";
            taskListModel.append({
                "taskName": txt,
                "checked": false,
                "index": taskListModel.count
            });
            inputTextArea.text = ""; // 清空輸入框
        }

        // 💡 Function B: 點擊已有按鈕（按下 / 重新命名）
        function handleButtonInteraction(targetButton, itemIndex) {
            var txt = inputTextArea.text.trim();
            if (txt !== "") {
                taskListModel.set(itemIndex, {
                    "taskName": txt,
                    "checked": !targetButton.checked
                });
                inputTextArea.text = "";
            } else {
                // 如果文字欄位沒字 ➡️ 執行按鈕原本的動作（例如切換選取狀態）
                taskListModel.set(itemIndex, {
                    "taskName": targetButton.buttonText,
                    "checked": !targetButton.checked
                });
            }
        }
        // 歸位動畫
        NumberAnimation on y {
            id: yAnimation
            running: false
            to: currentButton.index * (button_h + 10)
            duration: 200
        }
    }

    // 回覆用戶的文本
    component ContainerItem: Item {
        id: container
        property alias flowLayout: flowLayout
        property var textModel: "引頸期盼地等待訊息..."
        // 💡 1. 修改高度：外層 container 必須是「固定高度」或錨定到底部，滾輪才會生效
        clip: true
        // 設定實質邊界，Flow 才知道在哪裡折行
        //height: Math.max(button_h, flowLayout.height) // 高度完全由按鈕數量決定
        readonly property string customFont: "Courier"
        readonly property int customSize: 20

        // 💡 修正 1：主動監聽滑鼠滾輪事件，強制 Flickable 滾動
        MouseArea {
            anchors.fill: parent
            propagateComposedEvents: true // 允許事件傳下去
            // 點擊按下時，拒絕獨佔事件，強迫點擊穿透給底下的按鈕
            onPressed: mouse => mouse.accepted = false
            onWheel: wheel => {
                // 將滾輪力道除以 8 (120 / 8 = 15 像素)，這樣滑動幅度會非常溫和、平滑
                var scrollStep = wheel.angleDelta.y / 8;
                // 計算新的 Y 軸位置 (向上滾動時 scrollStep 是正的，contentY 要減少，所以用減法)
                var targetY = flowLayout.y + scrollStep;
                // 💡 限制滾動的上下邊界，避免文字飄出畫面外
                // 頂部邊界是 0，底部邊界是「可視高度 - 文字總高度」
                var minY = Math.min(0, container.height - flowLayout.height);
                flowLayout.y = Math.max(minY, Math.min(0, targetY));
            }
        }

        // 接收 Python 後端更新
        Connections {
            target: Backend

            function onResponseUpdated(all_text) {
                // 結尾加上換行符號並分割 ，避開重疊
                all_text = all_text + " @#@ ";
                container.textModel = all_text.split(" @#@ ");

                // 💡 核心修改：等待 QML 將新文字排版完畢後，自動滾動到最底部
                Qt.callLater(function () {
                    // 計算最底部的 y 座標（必須是負值或 0）
                    var bottomY = container.height - flowLayout.height;
                    // 如果文字總高度超過了容器可視高度，才需要滾動
                    if (bottomY < 0) {
                        flowLayout.y = bottomY;
                    } else {
                        flowLayout.y = 0; // 字數還很少時，維持在最頂端
                    }
                });
            }
        }
        // =========================================================================
        // 1. 底層文字（負責顯示咖啡色底色與抖動，Flow 會自動依文字寬度排列換行）
        // =========================================================================
        Flow {
            id: flowLayout
            //anchors.left: parent.left
            //anchors.right: parent.right
            //implicitHeight: childrenRect.height
            height: childrenRect.height
            width: container.width

            spacing: 1
            // 💡 4. 注意：在 Flickable 內部，左右邊界要錨定在 Flickable 的父層或固定寬度
            x: 0
            y: 0

            Repeater {
                model: container.textModel
                width: parent.width

                Text {
                    text: modelData // 這裡可以直接拿 modelData，完全不會報錯
                    width: container.width
                    wrapMode: Text.WrapAnywhere
                    horizontalAlignment: Text.AlignLeft
                    verticalAlignment: Text.AlignTop
                    color: Qt.rgba(0.68, 1, 0.18, 1.0)
                    font.family: container.customFont
                    font.pixelSize: container.customSize

                    MouseArea {
                        onClicked: mouse => {
                            // 點擊文字請求圖片列表
                            imgTimer.stop();       // 切換前先停止播放
                            Backend.getImages(modelData); // 傳文字給後端
                        }
                    }

                    // 特效層位移
                    transform: Translate {
                        // 💡 修正關鍵：使用 % 1.0 取餘數，讓字元位置在 0.0 ~ 1.0 之間無限循環
                        property real charPos: (index / 20.0) % 1.0

                        // 💡 修正 1：利用 index 計算出該字元專屬、固定不變的隨機值 (0.0 ~ 1.0)
                        // 這樣就不會在每一幀都重新隨機抽樣，能確保「固定某些字會動、某些字不會動」
                        property real charRandomSeed: (Math.sin(index * 12.9898) * 43758.5453) % 1.0

                        // 💡 修正 2：控制是否允許抖動的機率門檻（0.3 代表只有 30% 的字會動，其餘靜止）
                        property bool isSelectedToMove: Math.abs(charRandomSeed) < 0.4

                        x: {
                            var diff = neonFlow.pos - charPos;
                            if (diff < 0)
                                diff += 1.0;

                            // 光線經過時會降低震動，diff 同時增加震動機率和周長，手動調整光線要經過的時間，達成剛好經過觸發不少震動
                            // 同時必須滿足我們篩選出來的隨機字元（isSelectedToMove），才會產生位移
                            return (diff < 0.3 && isSelectedToMove) ? (Math.random() * 8 - 4) : 0;
                        }
                        y: {
                            var diff = neonFlow.pos - charPos;
                            if (diff < 0)
                                diff += 1.0;
                            return (diff < 0.3 && isSelectedToMove) ? (Math.random() * 4 - 2) : 0;
                        }
                    }
                }
            }
        }

        // =========================================================================
        // 2. 特效層（完美的 OpacityMask 霓虹燈效果）
        // =========================================================================
        Item {
            width: flowLayout.width
            height: flowLayout.height

            // 遮罩文字（結構必須與下方 flowLayout 完全對稱，包證換行位置 100% 貼合）
            Flow {
                id: maskFlow
                anchors.fill: parent
                spacing: 0
                visible: false // 必須隱藏，僅供遮罩裁切使用

                Repeater {
                    model: container.textModel
                    Text {
                        text: modelData
                        font.family: container.customFont
                        font.pixelSize: container.customSize
                    }
                }
            }

            // 霓虹斜線漸層
            LinearGradient {
                id: neonFlow
                anchors.fill: parent
                visible: false
                start: Qt.point(0, 0)
                end: Qt.point(parent.width * 0.5, parent.height)
                property real pos: -1.0
                //propagateComposedEvents: true // 允許事件傳下去

                gradient: Gradient {
                    GradientStop {
                        position: neonFlow.pos
                        color: "transparent"
                    }
                    GradientStop {
                        position: neonFlow.pos + 0.2
                        color: "#FF00FF"
                    }
                    GradientStop {
                        position: neonFlow.pos + 0.4
                        color: "transparent"
                    }
                    GradientStop {
                        position: neonFlow.pos + 0.3
                        color: "transparent"
                    }
                    GradientStop {
                        position: neonFlow.pos + 0.4
                        color: "#00FFFF"
                    }
                    GradientStop {
                        position: neonFlow.pos + 0.6
                        color: "transparent"
                    }
                    GradientStop {
                        position: neonFlow.pos + 0.5
                        color: "transparent"
                    }
                    GradientStop {
                        position: neonFlow.pos + 0.8
                        color: "#BF00FF"
                    }
                    GradientStop {
                        position: neonFlow.pos + 0.9
                        color: "transparent"
                    }
                }

                PropertyAnimation on pos {
                    from: -0.2
                    to: 11.0
                    duration: 8000 // 13/8 =1.4 適當
                    loops: Animation.Infinite
                    easing.type: Easing.Linear
                }
            }

            // 外發光
            Glow {
                id: neonGlow
                anchors.fill: parent
                source: neonFlow
                radius: 9
                samples: 19
                color: "#00FFFF"
                spread: 0.4
                visible: false
            }

            // 最終裁剪混合
            OpacityMask {
                anchors.fill: parent
                source: neonGlow
                maskSource: maskFlow
            }
        }
    }

    component InputBoxTextArea: TextArea {
        id: inputBox
        Layout.preferredHeight: Math.max(button_h, Math.min(contentHeight, window_h * 0.4))
        //height:Math.max(50, Math.min(contentHeight, window_h * 0.4))
        text: ""
        wrapMode: Text.Wrap // 自動換行
        placeholderText: "請輸入windowTittle, path, action... (:多重路徑、::分行、<>錄製)"
        focus: true // 點擊即可輸入
        // font.family: "Microsoft JhengHei" // 設置字體
        font.pixelSize: 18 // 設置字體大小
        color: Qt.rgba(0.1, 0.0, 0.15, 0.9)
        font.family: "Courier" // 用等寬字體更有駭客感

        background: Rectangle {
            //anchors.fill: parent
            width: parent.width
            height: parent.height
            color: Qt.rgba(0.68, 1, 0.18, 0.4)
            radius: 8
            border.color: Qt.rgba(0.68, 1, 0.18, 1)  // 深一點的邊框讓邊界更清晰
            border.width: 1
        }
        // 監聽文本變化
        onTextChanged: {
            if (text.length > 0) {
                // *** 進入 計算物體實際大小的 抓取模式
                if (!/^(.*)_W(\d+)_H(\d+)_Z([\d.]+)\.png$/.test(text)) {}
                // 當用戶輸入時更新 `userInput`
                userInput = text;
            }
        }
        Keys.onPressed: event => {
            // 當按下回車鍵時，執行提交操作
            if ([Qt.Key_Return, Qt.Key_Enter].includes(event.key)) {
                event.accepted = true;
                if (!(event.modifiers & (Qt.ShiftModifier | Qt.ControlModifier | Qt.AltModifier))) {
                    btn.animButton.clicked();
                    IC.input_line(userInput); // 執行失敗時同時不執行下一行

                    text = "";
                }
            }
        }
    }

    //WheelHandler {onWheel: root.scaleFactor += wheel.angleDelta.y * 0.001} 滾輪縮放
    component ColButton: Row {
        id: buttonContainer
        //anchors.top: parent.top
        // width: childrenRect.width
        // height: childrenRect.height
        //Layout.preferredWidth: button_w
        //Layout.preferredHeight: button_h
        spacing: 10 // 按鈕之間的間距
        property alias animButton: animButton //讓 btn.animButton.clicked() 抓到此id

        CamButton {
            id: animButton
            buttonText: "輸出動畫"

            onClicked: mouse => {
                // 拆字判斷動畫
                var target = keyword_map[userInput];
                if (!target) {
                    console.log("動畫❌ 找不到 userInput", userInput);
                    return;
                }
                var clipName = Object.keys(target)[0];
                // 搜尋 glTF 中的動畫列表
                for (var i = 0; i < iluluModel.animations.length; i++) {
                    var a = iluluModel.animations[i];
                    if (a.name === clipName) {
                        console.log("▶ 播放動畫:", clipName);
                        iluluModel.animations[i].position = a.start;
                        iluluModel.animations[i].duration = a.duration;
                        iluluModel.animations[i].running = true;
                        return;
                    } else if (a.name === "Idle") {
                        console.log("▶ 播放待機動畫:", clipName);
                        iluluModel.animations[i].position = a.start;
                        iluluModel.animations[i].duration = a.duration;
                        iluluModel.animations[i].running = true;
                    }
                }
            }
        }

        CamButton {
            id: quit
            buttonText: "關閉"

            onClicked: mouse => {
                // TODO:******** Gather TaskBarItem model data and send to backend for saving
                var users = [];
                for (var i = 0; i < taskBar.taskListModel.count; i++) {
                    var item = taskBar.taskListModel.get(i);
                    users.push({
                        "taskName": item.taskName,
                        "checked": item.checked,
                        "index": item.index
                    });
                }
                // Call backend to save user task buttons, then quit
                try {
                    Backend.saveUsers(users);
                } catch (e) {
                    console.log("保存使用者按鈕時發生錯誤:", e);
                }
                Qt.quit();
            }
        }
    }

    component ImgItem: Item {
        // 顯示回覆用戶的文本
        // TODO:收到的路徑轉成句子，
        // TODO:*** 打開路徑下的所有圖片
        // TODO:*** 點擊某個詞，顯示該詞對應路徑下的圖片集
        // TODO:*** 動態圖片：每次撥放幾張圖，循環撥放
        // pyside6-balsam *.glb 另存成 iluluModel.qml and meshes/*.mesh and maps/模型使用的貼圖
        Image {
            id: imgDisplay
            //width: 300
            //height: 300
            //anchors.fill: parent
            width: parent.width
            height: parent.height
            fillMode: Image.PreserveAspectFit
        }

        Timer {
            id: imgTimer
            interval: 1000 // 每秒切換
            repeat: true
            running: false
            onTriggered: {
                if (imageList.length > 0) {
                    imgDisplay.source = imageList[currentIndex];
                    currentIndex = (currentIndex + 1) % imageList.length;
                } else {
                    imgDisplay.source = ""; // 空列表清除圖片
                }
            }
        }

        property var imageList: []
        property int currentIndex: 0

        Connections {
            target: Backend
            function onImagesReady(images) {
                imageList = images;
                currentIndex = 0;
                if (imageList.length > 0) {
                    imgDisplay.source = imageList[0]; // 立即顯示第一張
                    imgTimer.start();
                } else {
                    imgDisplay.source = "";
                    imgTimer.stop();
                }
            }
        }
    }

    // == == == == UI：重新掛載子物件 == == == ==

    component RectAinmate: Rectangle {
        //anchors.right: parent.right
        //anchors.top: parent.top
        implicitWidth: 140
        implicitHeight: 140
        color: "#242424AA"
        radius: 8
        Column {
            //anchors.fill: parent
            //anchors.margins: 10
            width: parent.width
            height: parent.height
            spacing: 8
            Text {
                text: selectedNode ? "選取: " + selectedNode.objectName : "未選取節點"
                color: "white"
            }
            CamButton {
                buttonText: "掛到 iluluModel (root)"
                onClicked: mouse => {
                    if (selectedNode)
                        selectedNode.parent = iluluModel;
                }
            }

            CamButton {
                buttonText: "掛到 cam"

                // 關閉時文字黑剪影 像外凸，開啟時文字白影 像內凸

                onClicked: mouse => {
                    if (selectedNode)
                        selectedNode.parent = cam;
                }
            }
        }
    }
}
