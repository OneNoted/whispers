import QtQuick
import Quickshell
import Quickshell.Io
import Quickshell.Wayland

ShellRoot {
    id: root

    property string stage: "idle"
    property string detail: ""
    property string screenName: ""
    property bool shown: false
    property bool windowOpen: false

    readonly property color accent: {
        switch (stage) {
        case "listening": return "#ff625f";
        case "transcribing": return "#f5b942";
        case "enhancing": return "#8b8cff";
        case "delivering": return "#62c6ff";
        case "error": return "#ff625f";
        default: return "#54d68b";
        }
    }

    readonly property string label: {
        switch (stage) {
        case "listening": return "Listening";
        case "transcribing": return "Transcribing";
        case "enhancing": return "Enhancing";
        case "delivering": return "Typing";
        case "copied": return "Copied";
        case "error": return "Couldn't finish";
        case "info": return "Whispers";
        default: return "Done";
        }
    }

    readonly property bool finalStage: stage === "done"
        || stage === "copied"
        || stage === "error"
        || stage === "info"

    function present(nextStage, nextDetail, nextScreen) {
        stage = nextStage;
        detail = nextDetail;
        if (nextScreen.length > 0)
            screenName = nextScreen;
        closeTimer.stop();
        windowOpen = true;
        shown = true;
        if (finalStage)
            autoHideTimer.restart();
        else
            autoHideTimer.stop();
    }

    function dismiss() {
        autoHideTimer.stop();
        shown = false;
        closeTimer.restart();
    }

    IpcHandler {
        target: "whispers"

        function display(stage: string, detail: string, screen: string): string {
            root.present(stage, detail, screen);
            return "ok";
        }

        function hide(): string {
            root.dismiss();
            return "ok";
        }
    }

    Timer {
        id: autoHideTimer
        interval: root.stage === "error" ? 2600 : 1500
        repeat: false
        onTriggered: root.dismiss()
    }

    Timer {
        id: closeTimer
        interval: 190
        repeat: false
        onTriggered: {
            if (!root.shown)
                root.windowOpen = false;
        }
    }

    Variants {
        model: Quickshell.screens

        delegate: PanelWindow {
            id: window

            property var modelData: item
            readonly property bool selected: root.screenName.length > 0
                ? modelData.name === root.screenName
                : modelData === Quickshell.primaryScreen

            screen: modelData
            visible: root.windowOpen && selected
            color: "transparent"
            implicitWidth: 480
            implicitHeight: 92

            anchors {
                bottom: true
            }

            WlrLayershell.namespace: "whispers:osd"
            WlrLayershell.layer: WlrLayershell.Overlay
            WlrLayershell.exclusiveZone: -1
            WlrLayershell.keyboardFocus: WlrKeyboardFocus.None
            WlrLayershell.margins.bottom: 46

            Item {
                id: stageContainer
                anchors.centerIn: parent
                width: root.detail.length > 0 ? 430 : 226
                height: 56
                opacity: root.shown ? 1 : 0
                scale: root.shown ? 1 : 0.94

                Behavior on width {
                    NumberAnimation { duration: 170; easing.type: Easing.OutCubic }
                }
                Behavior on opacity {
                    NumberAnimation { duration: 140; easing.type: Easing.OutCubic }
                }
                Behavior on scale {
                    NumberAnimation { duration: 170; easing.type: Easing.OutBack }
                }

                Rectangle {
                    anchors.fill: pill
                    anchors.topMargin: 5
                    radius: pill.radius
                    color: "#70000000"
                }

                Rectangle {
                    id: pill
                    anchors.fill: parent
                    radius: 18
                    color: "#f2181a20"
                    border.width: 1
                    border.color: "#4a4e59"

                    Rectangle {
                        width: 3
                        height: parent.height - 18
                        radius: 2
                        color: root.accent
                        anchors.left: parent.left
                        anchors.leftMargin: 7
                        anchors.verticalCenter: parent.verticalCenter

                        Behavior on color { ColorAnimation { duration: 150 } }
                    }

                    Rectangle {
                        id: statusDisc
                        width: 34
                        height: 34
                        radius: 12
                        color: Qt.rgba(root.accent.r, root.accent.g, root.accent.b, 0.13)
                        anchors.left: parent.left
                        anchors.leftMargin: 20
                        anchors.verticalCenter: parent.verticalCenter

                        Behavior on color { ColorAnimation { duration: 150 } }

                        Item {
                            anchors.fill: parent
                            visible: !root.finalStage

                            Rectangle {
                                id: micHead
                                visible: root.stage === "listening"
                                width: 8
                                height: 14
                                radius: 4
                                color: "transparent"
                                border.width: 2
                                border.color: root.accent
                                anchors.horizontalCenter: parent.horizontalCenter
                                anchors.top: parent.top
                                anchors.topMargin: 6
                            }

                            Rectangle {
                                visible: root.stage === "listening"
                                width: 14
                                height: 10
                                radius: 7
                                color: "transparent"
                                border.width: 2
                                border.color: root.accent
                                anchors.horizontalCenter: parent.horizontalCenter
                                anchors.top: parent.top
                                anchors.topMargin: 12
                            }

                            Rectangle {
                                visible: root.stage === "listening"
                                width: 2
                                height: 5
                                radius: 1
                                color: root.accent
                                anchors.horizontalCenter: parent.horizontalCenter
                                anchors.top: parent.top
                                anchors.topMargin: 21
                            }

                            Rectangle {
                                visible: root.stage === "listening"
                                width: 10
                                height: 2
                                radius: 1
                                color: root.accent
                                anchors.horizontalCenter: parent.horizontalCenter
                                anchors.top: parent.top
                                anchors.topMargin: 25
                            }

                            Row {
                                visible: root.stage !== "listening"
                                spacing: 3
                                anchors.centerIn: parent

                                Repeater {
                                    model: 3

                                    Rectangle {
                                        property int barIndex: index
                                        width: 3
                                        height: 7
                                        radius: 2
                                        color: root.accent

                                        SequentialAnimation on height {
                                            running: root.shown && root.stage !== "listening" && !root.finalStage
                                            loops: Animation.Infinite
                                            PauseAnimation { duration: barIndex * 80 }
                                            NumberAnimation { to: 16; duration: 220; easing.type: Easing.InOutSine }
                                            NumberAnimation { to: 7; duration: 220; easing.type: Easing.InOutSine }
                                            PauseAnimation { duration: (2 - barIndex) * 80 }
                                        }
                                    }
                                }
                            }
                        }

                        Text {
                            visible: root.finalStage
                            anchors.centerIn: parent
                            text: root.stage === "error" ? "!" : "✓"
                            color: root.accent
                            font.pixelSize: root.stage === "error" ? 18 : 17
                            font.weight: Font.DemiBold
                        }
                    }

                    Row {
                        spacing: 10
                        anchors.left: statusDisc.right
                        anchors.leftMargin: 13
                        anchors.right: parent.right
                        anchors.rightMargin: 18
                        anchors.verticalCenter: parent.verticalCenter

                        Text {
                            text: root.label
                            color: "#f2f3f5"
                            font.pixelSize: 13
                            font.weight: Font.DemiBold
                            font.letterSpacing: 0.35
                            anchors.verticalCenter: parent.verticalCenter
                        }

                        Rectangle {
                            visible: root.detail.length > 0
                            width: 3
                            height: 3
                            radius: 2
                            color: "#686d78"
                            anchors.verticalCenter: parent.verticalCenter
                        }

                        Text {
                            visible: root.detail.length > 0
                            width: Math.max(0, parent.width - x)
                            text: root.detail
                            color: "#aeb2bd"
                            font.pixelSize: 12
                            elide: Text.ElideRight
                            maximumLineCount: 1
                            anchors.verticalCenter: parent.verticalCenter
                        }
                    }
                }
            }

            // Status only: never consume pointer input from the active app.
            mask: Region {}
        }
    }
}
