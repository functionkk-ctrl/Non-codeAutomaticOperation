import QtQuick
import QtQuick3D

Node {
    id: node

    // Resources
    property url textureData: "maps/textureData.png"
    property url textureData9: "maps/textureData9.png"
    property url textureData63: "maps/textureData63.png"
    property url textureData36: "maps/textureData36.png"
    property url textureData18: "maps/textureData18.png"
    property url textureData72: "maps/textureData72.png"
    property url textureData54: "maps/textureData54.png"
    property url textureData27: "maps/textureData27.png"
    property url textureData81: "maps/textureData81.png"
    property url textureData105: "maps/textureData105.png"
    property url textureData118: "maps/textureData118.png"
    property url textureData127: "maps/textureData127.png"
    property url textureData136: "maps/textureData136.png"
    property url textureData145: "maps/textureData145.png"
    Texture {
        id: _4_texture
        generateMipmaps: true
        mipFilter: Texture.Linear
        source: node.textureData
    }
    Texture {
        id: _0_texture
        generateMipmaps: true
        mipFilter: Texture.Linear
        source: node.textureData9
    }
    Texture {
        id: _6_texture
        generateMipmaps: true
        mipFilter: Texture.Linear
        source: node.textureData63
    }
    Texture {
        id: _3_texture
        generateMipmaps: true
        mipFilter: Texture.Linear
        source: node.textureData36
    }
    Texture {
        id: _1_texture
        generateMipmaps: true
        mipFilter: Texture.Linear
        source: node.textureData18
    }
    Texture {
        id: _7_texture
        generateMipmaps: true
        mipFilter: Texture.Linear
        source: node.textureData72
    }
    Texture {
        id: _5_texture
        generateMipmaps: true
        mipFilter: Texture.Linear
        source: node.textureData54
    }
    Texture {
        id: _2_texture
        generateMipmaps: true
        mipFilter: Texture.Linear
        source: node.textureData27
    }
    Texture {
        id: _8_texture
        generateMipmaps: true
        mipFilter: Texture.Linear
        source: node.textureData81
    }
    Texture {
        id: _9_texture
        generateMipmaps: true
        mipFilter: Texture.Linear
        source: node.textureData105
    }
    Texture {
        id: _10_texture
        generateMipmaps: true
        mipFilter: Texture.Linear
        source: node.textureData118
    }
    Texture {
        id: _11_texture
        generateMipmaps: true
        mipFilter: Texture.Linear
        source: node.textureData127
    }
    Texture {
        id: _12_texture
        generateMipmaps: true
        mipFilter: Texture.Linear
        source: node.textureData136
    }
    Texture {
        id: _13_texture
        generateMipmaps: true
        mipFilter: Texture.Linear
        source: node.textureData145
    }
    PrincipledMaterial {
        id: material_027_material
        objectName: "Material.027"
        baseColorMap: _3_texture
        roughness: 0.8999999761581421
        cullMode: PrincipledMaterial.NoCulling
        alphaMode: PrincipledMaterial.Opaque
        lighting: PrincipledMaterial.NoLighting
    }
    PrincipledMaterial {
        id: material_028_material
        objectName: "Material.028"
        baseColor: "#ff000000"
        roughness: 0.8999999761581421
        alphaMode: PrincipledMaterial.Opaque
        lighting: PrincipledMaterial.NoLighting
    }
    PrincipledMaterial {
        id: material_012_material
        objectName: "Material.012"
        baseColorMap: _4_texture
        roughness: 0.8999999761581421
        cullMode: PrincipledMaterial.NoCulling
        alphaMode: PrincipledMaterial.Opaque
        lighting: PrincipledMaterial.NoLighting
    }
    PrincipledMaterial {
        id: material_016_material
        objectName: "Material.016"
        baseColor: "#ff3d1d27"
        roughness: 0.8999999761581421
        alphaMode: PrincipledMaterial.Opaque
        lighting: PrincipledMaterial.NoLighting
    }
    PrincipledMaterial {
        id: material_005_material
        objectName: "Material.005"
        baseColorMap: _5_texture
        roughness: 0.8999999761581421
        cullMode: PrincipledMaterial.NoCulling
        alphaMode: PrincipledMaterial.Opaque
        lighting: PrincipledMaterial.NoLighting
    }
    PrincipledMaterial {
        id: material_031_material
        objectName: "Material.031"
        baseColor: "#ff000000"
        roughness: 0.8999999761581421
        alphaMode: PrincipledMaterial.Opaque
        lighting: PrincipledMaterial.NoLighting
    }
    PrincipledMaterial {
        id: material_006_material
        objectName: "Material.006"
        baseColorMap: _6_texture
        roughness: 0.8999999761581421
        cullMode: PrincipledMaterial.NoCulling
        alphaMode: PrincipledMaterial.Opaque
        lighting: PrincipledMaterial.NoLighting
    }
    PrincipledMaterial {
        id: material_032_material
        objectName: "Material.032"
        baseColor: "#ff000000"
        roughness: 0.8999999761581421
        alphaMode: PrincipledMaterial.Opaque
        lighting: PrincipledMaterial.NoLighting
    }
    PrincipledMaterial {
        id: material_015_material
        objectName: "Material.015"
        baseColorMap: _7_texture
        roughness: 0.8999999761581421
        alphaMode: PrincipledMaterial.Opaque
        lighting: PrincipledMaterial.NoLighting
    }
    PrincipledMaterial {
        id: material_019_material
        objectName: "Material.019"
        baseColor: "#ff603447"
        roughness: 0.8999999761581421
        alphaMode: PrincipledMaterial.Opaque
        lighting: PrincipledMaterial.NoLighting
    }
    PrincipledMaterial {
        id: material_014_material
        objectName: "Material.014"
        baseColorMap: _8_texture
        roughness: 0.8999999761581421
        cullMode: PrincipledMaterial.NoCulling
        alphaMode: PrincipledMaterial.Opaque
        lighting: PrincipledMaterial.NoLighting
    }
    PrincipledMaterial {
        id: material_002_material
        objectName: "Material.002"
        baseColorMap: _0_texture
        roughness: 0.8999999761581421
        cullMode: PrincipledMaterial.NoCulling
        alphaMode: PrincipledMaterial.Opaque
        lighting: PrincipledMaterial.NoLighting
    }
    PrincipledMaterial {
        id: material_018_material
        objectName: "Material.018"
        baseColor: "#ff603447"
        roughness: 0.8999999761581421
        alphaMode: PrincipledMaterial.Opaque
        lighting: PrincipledMaterial.NoLighting
    }
    PrincipledMaterial {
        id: material_011_material
        objectName: "Material.011"
        baseColor: "#ffe7e7e7"
        roughness: 0.8999999761581421
        cullMode: PrincipledMaterial.NoCulling
        alphaMode: PrincipledMaterial.Opaque
        lighting: PrincipledMaterial.NoLighting
    }
    PrincipledMaterial {
        id: material_024_material
        objectName: "Material.024"
        baseColor: "#ff000000"
        roughness: 0.8999999761581421
        cullMode: PrincipledMaterial.NoCulling
        alphaMode: PrincipledMaterial.Opaque
        lighting: PrincipledMaterial.NoLighting
    }
    PrincipledMaterial {
        id: material_007_material
        objectName: "Material.007"
        baseColor: "#ffe7e7e7"
        roughness: 0.8999999761581421
        cullMode: PrincipledMaterial.NoCulling
        alphaMode: PrincipledMaterial.Opaque
        lighting: PrincipledMaterial.NoLighting
    }
    PrincipledMaterial {
        id: material_029_material
        objectName: "Material.029"
        baseColorMap: _9_texture
        roughness: 0.8999999761581421
        cullMode: PrincipledMaterial.NoCulling
        alphaMode: PrincipledMaterial.Opaque
        lighting: PrincipledMaterial.NoLighting
    }
    PrincipledMaterial {
        id: material_021_material
        objectName: "Material.021"
        baseColor: "#ff000000"
        roughness: 0.8999999761581421
        alphaMode: PrincipledMaterial.Opaque
        lighting: PrincipledMaterial.NoLighting
    }
    PrincipledMaterial {
        id: material_030_material
        objectName: "Material.030"
        baseColor: "#ff000000"
        roughness: 0.8999999761581421
        alphaMode: PrincipledMaterial.Opaque
        lighting: PrincipledMaterial.NoLighting
    }
    PrincipledMaterial {
        id: material_010_material
        objectName: "Material.010"
        baseColor: "#ff000000"
        roughness: 0.8999999761581421
        cullMode: PrincipledMaterial.NoCulling
        alphaMode: PrincipledMaterial.Opaque
        lighting: PrincipledMaterial.NoLighting
    }
    PrincipledMaterial {
        id: material_008_material
        objectName: "Material.008"
        baseColorMap: _10_texture
        roughness: 0.8999999761581421
        cullMode: PrincipledMaterial.NoCulling
        alphaMode: PrincipledMaterial.Opaque
        lighting: PrincipledMaterial.NoLighting
    }
    PrincipledMaterial {
        id: material_003_material
        objectName: "Material.003"
        baseColorMap: _1_texture
        roughness: 0.8999999761581421
        cullMode: PrincipledMaterial.NoCulling
        alphaMode: PrincipledMaterial.Opaque
        lighting: PrincipledMaterial.NoLighting
    }
    PrincipledMaterial {
        id: material_020_material
        objectName: "Material.020"
        baseColor: "#ff5f3c3a"
        roughness: 0.8999999761581421
        alphaMode: PrincipledMaterial.Opaque
        lighting: PrincipledMaterial.NoLighting
    }
    PrincipledMaterial {
        id: material_009_material
        objectName: "Material.009"
        baseColorMap: _11_texture
        roughness: 0.8999999761581421
        cullMode: PrincipledMaterial.NoCulling
        alphaMode: PrincipledMaterial.Opaque
        lighting: PrincipledMaterial.NoLighting
    }
    PrincipledMaterial {
        id: material_022_material
        objectName: "Material.022"
        baseColor: "#ff000000"
        roughness: 0.8999999761581421
        alphaMode: PrincipledMaterial.Opaque
        lighting: PrincipledMaterial.NoLighting
    }
    PrincipledMaterial {
        id: material_023_material
        objectName: "Material.023"
        baseColor: "#ff000000"
        roughness: 0.8999999761581421
        alphaMode: PrincipledMaterial.Opaque
        lighting: PrincipledMaterial.NoLighting
    }
    PrincipledMaterial {
        id: material_026_material
        objectName: "Material.026"
        baseColorMap: _12_texture
        roughness: 0.8999999761581421
        cullMode: PrincipledMaterial.NoCulling
        alphaMode: PrincipledMaterial.Opaque
        lighting: PrincipledMaterial.NoLighting
    }
    PrincipledMaterial {
        id: material_004_material
        objectName: "Material.004"
        baseColorMap: _2_texture
        roughness: 0.8999999761581421
        cullMode: PrincipledMaterial.NoCulling
        alphaMode: PrincipledMaterial.Opaque
        lighting: PrincipledMaterial.NoLighting
    }
    PrincipledMaterial {
        id: material_033_material
        objectName: "Material.033"
        baseColor: "#ff674245"
        roughness: 0.8999999761581421
        alphaMode: PrincipledMaterial.Opaque
        lighting: PrincipledMaterial.NoLighting
    }
    PrincipledMaterial {
        id: material_013_material
        objectName: "Material.013"
        baseColorMap: _13_texture
        roughness: 0.8999999761581421
        cullMode: PrincipledMaterial.NoCulling
        alphaMode: PrincipledMaterial.Opaque
        lighting: PrincipledMaterial.NoLighting
    }
    PrincipledMaterial {
        id: material_025_material
        objectName: "Material.025"
        baseColor: "#ff775349"
        roughness: 0.8999999761581421
        alphaMode: PrincipledMaterial.Opaque
        lighting: PrincipledMaterial.NoLighting
    }
    PrincipledMaterial {
        id: material_017_material
        objectName: "Material.017"
        baseColor: "#ff000000"
        roughness: 0.8999999761581421
        alphaMode: PrincipledMaterial.Opaque
        lighting: PrincipledMaterial.NoLighting
    }

    // Nodes:
    Node {
        id: sketchfab_model
        objectName: "Sketchfab_model"
        rotation: Qt.quaternion(0.707107, -0.707107, 0, 0)
        scale: Qt.vector3d(1, 1, 1)
        Node {
            id: f891fd68ddec4743a73162f3e970d831_fbx
            objectName: "f891fd68ddec4743a73162f3e970d831.fbx"
            rotation: Qt.quaternion(0.707107, 0.707107, 0, 0)
            scale: Qt.vector3d(0.01, 0.01, 0.01)
            Node {
                id: rootNode
                objectName: "RootNode"
                Node {
                    id: camisa
                    objectName: "camisa"
                    position: Qt.vector3d(0, 0, 26.7288)
                    rotation: Qt.quaternion(0.707107, -0.707107, 0, 0)
                    scale: Qt.vector3d(100, 100, 100)
                    Model {
                        id: camisa_Material_001_0
                        objectName: "camisa_Material.001_0"
                        source: "meshes/camisa_Material_001_0_mesh.mesh"
                        pickable: true // 觸碰模型

                        materials: [
                            material_002_material
                        ]
                    }
                    Model {
                        id: camisa_Material_021_0
                        objectName: "camisa_Material.021_0"
                        source: "meshes/camisa_Material_021_0_mesh.mesh"
                        pickable: true // 觸碰模型

                        materials: [
                            material_021_material
                        ]
                    }
                }
                Node {
                    id: cylinder
                    objectName: "Cylinder"
                    rotation: Qt.quaternion(0.707107, -0.707107, 0, 0)
                    scale: Qt.vector3d(100, 100, 100)
                    Model {
                        id: cylinder_Material_002_0
                        objectName: "Cylinder_Material.002_0"
                        source: "meshes/cylinder_Material_002_0_mesh.mesh"
                        pickable: true // 觸碰模型

                        materials: [
                            material_003_material
                        ]
                    }
                    Model {
                        id: cylinder_Material_022_0
                        objectName: "Cylinder_Material.022_0"
                        source: "meshes/cylinder_Material_022_0_mesh.mesh"
                        pickable: true // 觸碰模型

                        materials: [
                            material_022_material
                        ]
                    }
                }
                Node {
                    id: cylinder_001
                    objectName: "Cylinder.001"
                    position: Qt.vector3d(0, 249.134, 0)
                    rotation: Qt.quaternion(0.707106, -0.707107, 0, 0)
                    scale: Qt.vector3d(7.12122, 7.12122, 7.12122)
                    Model {
                        id: cylinder_001_Material_003_0
                        objectName: "Cylinder.001_Material.003_0"
                        source: "meshes/cylinder_001_Material_003_0_mesh.mesh"
                        pickable: true // 觸碰模型

                        materials: [
                            material_004_material
                        ]
                    }
                    Model {
                        id: cylinder_001_Material_025_0
                        objectName: "Cylinder.001_Material.025_0"
                        source: "meshes/cylinder_001_Material_025_0_mesh.mesh"
                        pickable: true // 觸碰模型

                        materials: [
                            material_025_material
                        ]
                    }
                }
                Node {
                    id: cylinder_003
                    objectName: "Cylinder.003"
                    position: Qt.vector3d(0, 0.573945, 3.73064)
                    rotation: Qt.quaternion(0.707107, -0.707107, 0, 0)
                    scale: Qt.vector3d(100, 100, 100)
                    Model {
                        id: cylinder_003_Material_027_0
                        objectName: "Cylinder.003_Material.027_0"
                        source: "meshes/cylinder_003_Material_027_0_mesh.mesh"
                        pickable: true // 觸碰模型

                        materials: [
                            material_027_material
                        ]
                    }
                    Model {
                        id: cylinder_003_Material_028_0
                        objectName: "Cylinder.003_Material.028_0"
                        source: "meshes/cylinder_003_Material_028_0_mesh.mesh"
                        pickable: true // 觸碰模型

                        materials: [
                            material_028_material
                        ]
                    }
                }
                Node {
                    id: nurbsPath
                    objectName: "NurbsPath"
                    position: Qt.vector3d(20.5484, 285.587, 3.86377e-05)
                    rotation: Qt.quaternion(0.53354, -0.501863, 0.495883, -0.466442)
                    scale: Qt.vector3d(10.7615, 10.7615, 10.7615)
                    Model {
                        id: nurbsPath_Material_012_0
                        objectName: "NurbsPath_Material.012_0"
                        source: "meshes/nurbsPath_Material_012_0_mesh.mesh"
                        pickable: true // 觸碰模型

                        materials: [
                            material_012_material
                        ]
                    }
                    Model {
                        id: nurbsPath_Material_016_0
                        objectName: "NurbsPath_Material.016_0"
                        source: "meshes/nurbsPath_Material_016_0_mesh.mesh"
                        pickable: true // 觸碰模型

                        materials: [
                            material_016_material
                        ]
                    }
                }
                Node {
                    id: nurbsPath_001
                    objectName: "NurbsPath.001"
                    position: Qt.vector3d(-81.2172, 281.41, 3.80766e-05)
                    rotation: Qt.quaternion(0.517944, -0.517944, 0.481388, -0.481388)
                    scale: Qt.vector3d(10.7615, 10.7615, 10.7615)
                    Model {
                        id: nurbsPath_001_Material_005_0
                        objectName: "NurbsPath.001_Material.005_0"
                        position: Qt.vector3d(-4.76837e-06, 0, 0)
                        source: "meshes/nurbsPath_001_Material_005_0_mesh.mesh"
                        pickable: true // 觸碰模型

                        materials: [
                            material_005_material
                        ]
                    }
                    Model {
                        id: nurbsPath_001_Material_031_0
                        objectName: "NurbsPath.001_Material.031_0"
                        position: Qt.vector3d(-4.76837e-06, 0, 0)
                        source: "meshes/nurbsPath_001_Material_031_0_mesh.mesh"
                        pickable: true // 觸碰模型

                        materials: [
                            material_031_material
                        ]
                    }
                }
                Node {
                    id: nurbsPath_002
                    objectName: "NurbsPath.002"
                    position: Qt.vector3d(20.5484, 285.587, 3.86377e-05)
                    rotation: Qt.quaternion(0.517944, -0.517944, 0.481388, -0.481388)
                    scale: Qt.vector3d(10.7615, 10.7615, 10.7615)
                    Model {
                        id: nurbsPath_002_Material_006_0
                        objectName: "NurbsPath.002_Material.006_0"
                        source: "meshes/nurbsPath_002_Material_006_0_mesh.mesh"
                        pickable: true // 觸碰模型

                        materials: [
                            material_006_material
                        ]
                    }
                    Model {
                        id: nurbsPath_002_Material_032_0
                        objectName: "NurbsPath.002_Material.032_0"
                        source: "meshes/nurbsPath_002_Material_032_0_mesh.mesh"
                        pickable: true // 觸碰模型

                        materials: [
                            material_032_material
                        ]
                    }
                }
                Node {
                    id: nurbsPath_003
                    objectName: "NurbsPath.003"
                    position: Qt.vector3d(20.0227, 289.876, 2.83878)
                    rotation: Qt.quaternion(0.567125, -0.518934, 0.476574, -0.42656)
                    scale: Qt.vector3d(12.8823, 12.1917, 12.1886)
                    Model {
                        id: nurbsPath_003_Material_015_0
                        objectName: "NurbsPath.003_Material.015_0"
                        source: "meshes/nurbsPath_003_Material_015_0_mesh.mesh"
                        pickable: true // 觸碰模型

                        materials: [
                            material_015_material
                        ]
                    }
                    Model {
                        id: nurbsPath_003_Material_019_0
                        objectName: "NurbsPath.003_Material.019_0"
                        source: "meshes/nurbsPath_003_Material_019_0_mesh.mesh"
                        pickable: true // 觸碰模型

                        materials: [
                            material_019_material
                        ]
                    }
                }
                Node {
                    id: nurbsPath_005
                    objectName: "NurbsPath.005"
                    position: Qt.vector3d(-7.5773, 286.956, -1.47611)
                    rotation: Qt.quaternion(0.447793, -0.513575, 0.616934, -0.393846)
                    scale: Qt.vector3d(12.8854, 12.2374, 12.7089)
                    Model {
                        id: nurbsPath_005_Material_014_0
                        objectName: "NurbsPath.005_Material.014_0"
                        position: Qt.vector3d(4.74229e-06, 0, 0)
                        pickable: true // 觸碰模型

                        source: "meshes/nurbsPath_005_Material_014_0_mesh.mesh"
                        materials: [
                            material_014_material
                        ]
                    }
                    Model {
                        id: nurbsPath_005_Material_014_0_001
                        objectName: "NurbsPath.005_Material.014_0.001"
                        position: Qt.vector3d(4.74229e-06, 0, 0)
                        source: "meshes/nurbsPath_005_Material_014_0_001_mesh.mesh"
                        pickable: true // 觸碰模型

                        materials: [
                            material_014_material
                        ]
                    }
                    Model {
                        id: nurbsPath_005_Material_018_0
                        objectName: "NurbsPath.005_Material.018_0"
                        position: Qt.vector3d(4.74229e-06, 0, 0)
                        source: "meshes/nurbsPath_005_Material_018_0_mesh.mesh"
                        pickable: true // 觸碰模型

                        materials: [
                            material_018_material
                        ]
                    }
                    Model {
                        id: nurbsPath_005_Material_018_0_001
                        objectName: "NurbsPath.005_Material.018_0.001"
                        position: Qt.vector3d(4.74229e-06, 0, 0)
                        source: "meshes/nurbsPath_005_Material_018_0_001_mesh.mesh"
                        pickable: true // 觸碰模型

                        materials: [
                            material_018_material
                        ]
                    }
                }
                Node {
                    id: plane
                    objectName: "Plane"
                    position: Qt.vector3d(0, 233.419, -42.1153)
                    rotation: Qt.quaternion(0.00489896, 0.999988, 0, 0)
                    scale: Qt.vector3d(6.01423, 6.01423, 6.01423)
                    Model {
                        id: plane_Material_011_0
                        objectName: "Plane_Material.011_0"
                        position: Qt.vector3d(0, -6.68317e-06, 0)
                        source: "meshes/plane_Material_011_0_mesh.mesh"
                        pickable: true // 觸碰模型

                        materials: [
                            material_011_material
                        ]
                    }
                    Model {
                        id: plane_Material_024_0
                        objectName: "Plane_Material.024_0"
                        position: Qt.vector3d(0, -6.68317e-06, 0)
                        source: "meshes/plane_Material_024_0_mesh.mesh"
                        pickable: true // 觸碰模型

                        materials: [
                            material_024_material
                        ]
                    }
                }
                Node {
                    id: plane_001
                    objectName: "Plane.001"
                    position: Qt.vector3d(0, 316.824, 11.2553)
                    rotation: Qt.quaternion(0.995456, -0.0952267, 0, 0)
                    scale: Qt.vector3d(49.2248, 49.2248, 49.2248)
                    Model {
                        id: plane_001_Material_007_0
                        objectName: "Plane.001_Material.007_0"
                        source: "meshes/plane_001_Material_007_0_mesh.mesh"
                        pickable: true // 觸碰模型

                        materials: [
                            material_007_material
                        ]
                    }
                }
                Node {
                    id: plane_002
                    objectName: "Plane.002"
                    rotation: Qt.quaternion(0.707107, -0.707107, 0, 0)
                    scale: Qt.vector3d(100, 100, 100)
                    Model {
                        id: plane_002_Material_029_0
                        objectName: "Plane.002_Material.029_0"
                        source: "meshes/plane_002_Material_029_0_mesh.mesh"
                        pickable: true // 觸碰模型

                        materials: [
                            material_029_material
                        ]
                    }
                    Model {
                        id: plane_002_Material_030_0
                        objectName: "Plane.002_Material.030_0"
                        source: "meshes/plane_002_Material_030_0_mesh.mesh"
                        pickable: true // 觸碰模型

                        materials: [
                            material_030_material
                        ]
                    }
                }
                Node {
                    id: plane_003
                    objectName: "Plane.003"
                    position: Qt.vector3d(0, 197.675, 25.082)
                    rotation: Qt.quaternion(0.987372, -0.158419, 0, 0)
                    scale: Qt.vector3d(20.5228, 20.5227, 20.5227)
                    Model {
                        id: plane_003_Material_010_0
                        objectName: "Plane.003_Material.010_0"
                        source: "meshes/plane_003_Material_010_0_mesh.mesh"
                        pickable: true // 觸碰模型

                        materials: [
                            material_010_material
                        ]
                    }
                }
                Node {
                    id: roundcube_002
                    objectName: "Roundcube.002"
                    position: Qt.vector3d(0, 188.285, 0)
                    rotation: Qt.quaternion(0.707107, -0.707107, 0, 0)
                    scale: Qt.vector3d(24.324, 24.324, 24.324)
                    Model {
                        id: roundcube_002_Material_008_0
                        objectName: "Roundcube.002_Material.008_0"
                        source: "meshes/roundcube_002_Material_008_0_mesh.mesh"
                        pickable: true // 觸碰模型

                        materials: [
                            material_008_material
                        ]
                    }
                    Model {
                        id: roundcube_002_Material_020_0
                        objectName: "Roundcube.002_Material.020_0"
                        source: "meshes/roundcube_002_Material_020_0_mesh.mesh"
                        pickable: true // 觸碰模型

                        materials: [
                            material_020_material
                        ]
                    }
                }
                Node {
                    id: roundcube_003
                    objectName: "Roundcube.003"
                    position: Qt.vector3d(0, 277.759, 0)
                    rotation: Qt.quaternion(0.707107, -0.707107, 0, 0)
                    scale: Qt.vector3d(16.0427, 16.0427, 16.0427)
                    Model {
                        id: roundcube_003_Material_004_0
                        objectName: "Roundcube.003_Material.004_0"
                        source: "meshes/roundcube_003_Material_004_0_mesh.mesh"
                        pickable: true // 觸碰模型

                        materials: [
                            material_009_material
                        ]
                    }
                    Model {
                        id: roundcube_003_Material_023_0
                        objectName: "Roundcube.003_Material.023_0"
                        source: "meshes/roundcube_003_Material_023_0_mesh.mesh"
                        pickable: true // 觸碰模型

                        materials: [
                            material_023_material
                        ]
                    }
                }
                Node {
                    id: roundcube_005
                    objectName: "Roundcube.005"
                    position: Qt.vector3d(0, 0, 21.819)
                    rotation: Qt.quaternion(0.707107, -0.707107, 0, 0)
                    scale: Qt.vector3d(100, 100, 100)
                    Model {
                        id: roundcube_005_Material_026_0
                        objectName: "Roundcube.005_Material.026_0"
                        source: "meshes/roundcube_005_Material_026_0_mesh.mesh"
                        pickable: true // 觸碰模型

                        materials: [
                            material_026_material
                        ]
                    }
                    Model {
                        id: roundcube_005_Material_033_0
                        objectName: "Roundcube.005_Material.033_0"
                        source: "meshes/roundcube_005_Material_033_0_mesh.mesh"
                        pickable: true // 觸碰模型

                        materials: [
                            material_033_material
                        ]
                    }
                }
                Node {
                    id: roundcube_006
                    objectName: "Roundcube.006"
                    position: Qt.vector3d(0, 286.274, 0)
                    rotation: Qt.quaternion(0.707107, -0.707107, 0, 0)
                    scale: Qt.vector3d(38.6378, 38.6378, 38.6378)
                    Model {
                        id: roundcube_006_Material_013_0
                        objectName: "Roundcube.006_Material.013_0"
                        source: "meshes/roundcube_006_Material_013_0_mesh.mesh"
                        pickable: true // 觸碰模型

                        materials: [
                            material_013_material
                        ]
                    }
                    Model {
                        id: roundcube_006_Material_017_0
                        objectName: "Roundcube.006_Material.017_0"
                        source: "meshes/roundcube_006_Material_017_0_mesh.mesh"
                        pickable: true // 觸碰模型

                        materials: [
                            material_017_material
                        ]
                    }
                }
            }
        }
    }

    // Animations:
}
