from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "熊文钦_52252313128_人工智能.docx"
FIG = ROOT / "data" / "figures"


def set_font(run, size=12, bold=False, color=None, name="宋体"):
    run.font.name = name
    run._element.rPr.rFonts.set(qn("w:eastAsia"), name)
    run.font.size = Pt(size)
    run.bold = bold
    if color:
        run.font.color.rgb = RGBColor(*color)


def add_paragraph(doc, text="", size=12, bold=False, align=None, first_line=True, spacing_after=6):
    p = doc.add_paragraph()
    p.paragraph_format.line_spacing = 1.5
    p.paragraph_format.space_after = Pt(spacing_after)
    if first_line:
        p.paragraph_format.first_line_indent = Pt(24)
    if align is not None:
        p.alignment = align
    r = p.add_run(text)
    set_font(r, size=size, bold=bold)
    return p


def add_heading(doc, text, level=1):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(10 if level == 1 else 6)
    p.paragraph_format.space_after = Pt(6)
    p.paragraph_format.line_spacing = 1.3
    if level == 1:
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        size = 16
    else:
        p.alignment = WD_ALIGN_PARAGRAPH.LEFT
        size = 14 if level == 2 else 12
    r = p.add_run(text)
    set_font(r, size=size, bold=True, name="黑体")
    return p


def shade_cell(cell, fill):
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:fill"), fill)
    tc_pr.append(shd)


def set_cell_text(cell, text, bold=False, align=WD_ALIGN_PARAGRAPH.CENTER, size=10.5):
    cell.text = ""
    p = cell.paragraphs[0]
    p.alignment = align
    p.paragraph_format.line_spacing = 1.2
    r = p.add_run(text)
    set_font(r, size=size, bold=bold)
    cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER


def add_table(doc, headers, rows, widths=None):
    table = doc.add_table(rows=1, cols=len(headers))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.style = "Table Grid"
    for i, h in enumerate(headers):
        set_cell_text(table.rows[0].cells[i], h, bold=True)
        shade_cell(table.rows[0].cells[i], "D9EAF7")
    for row in rows:
        cells = table.add_row().cells
        for i, value in enumerate(row):
            align = WD_ALIGN_PARAGRAPH.LEFT if len(str(value)) > 12 else WD_ALIGN_PARAGRAPH.CENTER
            set_cell_text(cells[i], str(value), align=align)
    if widths:
        for row in table.rows:
            for i, width in enumerate(widths):
                row.cells[i].width = Cm(width)
    doc.add_paragraph()
    return table


def add_caption(doc, text):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(8)
    r = p.add_run(text)
    set_font(r, size=10.5, bold=False)


def add_image(doc, filename, width_cm, caption):
    path = FIG / filename
    if path.exists():
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.add_run().add_picture(str(path), width=Cm(width_cm))
        add_caption(doc, caption)


def cover(doc):
    for section in doc.sections:
        section.top_margin = Cm(2.6)
        section.bottom_margin = Cm(2.4)
        section.left_margin = Cm(2.8)
        section.right_margin = Cm(2.6)

    add_paragraph(doc, "重庆理工大学课程论文", size=22, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, first_line=False, spacing_after=40)
    add_paragraph(doc, "视频驱动三维可视化与智能编辑系统设计与实现", size=20, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, first_line=False, spacing_after=56)

    info = [
        ("课  程 名 称", "计算机视觉"),
        ("学        号", "52252313128"),
        ("研   究   生", "熊文钦"),
        ("电        话", "未填写"),
        ("课  程 教 师", "王海琨"),
        ("学  科 专 业", "人工智能"),
        ("研  究 方 向", "计算机视觉与三维重建"),
        ("培  养 单 位", "重庆理工大学"),
        ("完  成 时 间", "2026年5月18日"),
    ]
    table = doc.add_table(rows=len(info), cols=2)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, (k, v) in enumerate(info):
        set_cell_text(table.cell(i, 0), f"{k}：", bold=True, align=WD_ALIGN_PARAGRAPH.RIGHT, size=12)
        set_cell_text(table.cell(i, 1), v, align=WD_ALIGN_PARAGRAPH.LEFT, size=12)
        table.cell(i, 0).width = Cm(5)
        table.cell(i, 1).width = Cm(8)
    for row in table.rows:
        for cell in row.cells:
            tc_pr = cell._tc.get_or_add_tcPr()
            borders = OxmlElement("w:tcBorders")
            for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
                tag = OxmlElement(f"w:{edge}")
                tag.set(qn("w:val"), "nil")
                borders.append(tag)
            tc_pr.append(borders)
    doc.add_page_break()


def abstract_pages(doc):
    add_heading(doc, "摘  要", 1)
    add_paragraph(doc, "本文围绕计算机视觉课程中“视频驱动三维可视化与智能编辑系统”的综合实践要求，设计并实现了一套从视频输入、三维模型生成、模型导入导出、三维实例理解到材质与光照编辑的端到端系统。系统以 Python 为主要开发语言，使用 OpenCV 完成视频解码、帧采样和基础预处理，使用 PyQt5 与 OpenGL 构建可交互界面，使用点云、网格和神经网络模块组织三维场景处理流程。针对原始单目视频重建结果稀疏、形体不稳定的问题，本文将最终重建链路改为基于 S3DIS 室内点云数据的 RGB-D 推进视频生成与 Open3D TSDF 融合：先从带语义标注的室内场景生成稳定的彩色视频、深度图和相机位姿，再将 RGB-D 序列融合为三角网格。该方法牺牲了对任意手持单目视频的完全泛化能力，但显著提高了课程演示中“从视频到完整室内场景模型”的稳定性和可检查性。", spacing_after=4)
    add_paragraph(doc, "系统的三维理解部分包含实例识别和实例分割两个层次。实例识别模块采用 PointNet++/VoteNet 思想对点云采样结果进行候选框与类别预测；实例分割模块采用 PointGroup 风格的点特征编码、语义预测和空间聚类，将重建场景划分为可独立选择和编辑的物体单元。为满足课程对自主训练的要求，本文选择三维语义分割模块作为自主训练对象，基于整理后的室内点云数据集进行训练，保存了 epoch 10 的模型检查点，并在系统启动时自动加载该检查点。材质编辑模块预置金属、木质、塑料、玻璃、石材、陶瓷和织物七类材质；光照模块支持环境光、方向光、点光源和聚光灯参数管理。实验结果表明，系统能够完成“视频导入、三维生成、模型导出、外部模型导入、实例识别、实例分割、材质替换、光照调节、结果显示”的连续演示流程。", spacing_after=4)
    add_paragraph(doc, "关键词：计算机视觉；三维重建；视频驱动；实例分割；材质编辑", first_line=False)
    doc.add_page_break()

    add_heading(doc, "Abstract", 1)
    add_paragraph(doc, "This paper presents a video-driven 3D visualization and intelligent editing system for the Computer Vision course project. The system covers an end-to-end workflow from video import, 3D reconstruction, model import and export, 3D scene understanding, to material and lighting editing. OpenCV is used for video decoding and frame processing, PyQt5 and OpenGL are used for the interactive interface, and point-cloud/mesh representations are used to connect reconstruction and editing modules. To improve the unstable reconstruction quality of the original monocular prototype, the final reconstruction workflow uses an S3DIS indoor point cloud to generate a forward RGB-D walkthrough sequence, then fuses color, depth and camera poses with Open3D Scalable TSDF Fusion.", first_line=False)
    add_paragraph(doc, "For 3D understanding, the system includes instance detection and instance segmentation. The detection module follows the PointNet++ and VoteNet design, while the segmentation module follows a PointGroup-style pipeline with point feature encoding, semantic prediction and spatial clustering. To satisfy the requirement of independent training, the 3D segmentation model is trained from the prepared indoor point-cloud dataset, and the saved epoch-10 checkpoint is loaded by the GUI. Seven materials and four lighting types are implemented for scene editing. Experiments show that the system can demonstrate the complete workflow required by the course project.", first_line=False)
    add_paragraph(doc, "Keywords: computer vision; 3D reconstruction; video-driven system; instance segmentation; material editing", first_line=False)
    doc.add_page_break()


def toc(doc):
    add_heading(doc, "目  录", 1)
    entries = [
        ("摘要", "II"), ("Abstract", "III"),
        ("1 系统设计", "1"), ("1.1 需求分析", "1"), ("1.2 总体架构", "2"), ("1.3 端到端工作流", "3"),
        ("2 算法设计", "4"), ("2.1 视频导入与预处理", "4"), ("2.2 视频到三维模型生成", "5"), ("2.3 三维理解算法", "7"), ("2.4 材质与光照编辑", "8"),
        ("3 自主训练过程", "9"), ("4 系统实现", "11"), ("5 结果分析与结论", "13"), ("致谢", "15"), ("参考文献", "16"),
    ]
    for title, page in entries:
        p = doc.add_paragraph()
        p.paragraph_format.space_after = Pt(1)
        r = p.add_run(f"{title}{'.' * max(4, 48 - len(title) * 2)}{page}")
        set_font(r, size=12)
    doc.add_page_break()


def body(doc):
    add_heading(doc, "1 系统设计", 1)
    add_heading(doc, "1.1 需求分析", 2)
    add_paragraph(doc, "课程要求系统能够完成从视频输入到三维模型生成、三维理解与场景编辑的完整流程，并且要求至少一个模块包含完全自主训练过程。结合这一目标，本文将系统拆分为视频处理、三维重建、模型 I/O、实例识别、实例分割、材质编辑、光照控制、图形界面和训练九个模块。各模块既可通过 GUI 连贯调用，也可通过脚本单独复现实验结果。")
    add_table(doc, ["序号", "课程要求", "系统实现位置", "完成情况"], [
        ("1", "视频导入与信息读取", "core/video_processor.py, gui/video_panel.py", "完成"),
        ("2", "从视频生成三维模型", "tools/generate_s3dis_scene_video.py, core/rgbd_tsdf_reconstructor.py", "完成"),
        ("3", "三维模型导出", "core/model_io.py, data/output_models/*.ply/*.obj", "完成"),
        ("4", "外部模型导入", "core/model_io.py, gui/main_window.py", "完成"),
        ("5", "三维实例识别", "core/instance_detector.py", "完成"),
        ("6", "三维实例分割", "core/instance_segmentor.py", "完成"),
        ("7", "5种以上材质替换", "core/material_editor.py", "7种材质"),
        ("8", "光照添加与调节", "core/lighting_system.py", "4类光源"),
        ("9", "自主训练模块", "training/train_segmentation.py, data/checkpoints", "epoch 10 检查点"),
    ], widths=[1.2, 4.2, 6.2, 2.2])

    add_heading(doc, "1.2 总体架构", 2)
    add_paragraph(doc, "系统采用分层模块化结构。界面层负责视频面板、三维视口、控制面板和场景树；业务层负责把用户操作转化为视频处理、重建、识别、分割和编辑任务；算法层封装神经网络、几何处理、模型格式转换与训练流程。这样的设计使每个模块可以独立测试，也便于将重建算法从原先的不稳定单目视频方案替换为更稳定的 S3DIS RGB-D TSDF 融合方案。")
    add_table(doc, ["层次", "主要模块", "说明"], [
        ("界面层", "gui/main_window.py, viewport_3d.py, control_panel.py", "提供完整点击演示入口和三维显示窗口"),
        ("业务层", "core/video_processor.py, model_io.py", "组织数据流、文件读写和状态更新"),
        ("算法层", "core/rgbd_tsdf_reconstructor.py, instance_segmentor.py, tools/generate_s3dis_scene_video.py", "完成三维重建、识别、分割和结果生成"),
        ("训练层", "training/train_segmentation.py, models/segmentation_model.py", "自主训练 PointNet++ 风格分割模型"),
    ], widths=[2.2, 6.8, 5.0])

    add_heading(doc, "1.3 端到端工作流", 2)
    add_paragraph(doc, "完整演示流程为：从 S3DIS 格式室内点云生成一段由一个位置平滑推进到另一位置的 MP4 视频，同时保存对应的 color、depth、pose 与相机内参；在 GUI 中导入该视频并执行 3D Reconstruction；系统自动查找同名 RGB-D 工程，使用 TSDF 融合恢复室内场景网格；随后加载 S3DIS 语义对象到 Scene Objects 列表，用户可选择对象进行高亮、材质替换或右键删除；最后调节光照并导出模型。该流程覆盖课程文档列出的所有功能节点。")
    add_image(doc, "video_frame_00.png", 9.0, "图1 输入 S3DIS 推进视频的第1帧")

    add_heading(doc, "2 算法设计", 1)
    add_heading(doc, "2.1 视频导入与预处理", 2)
    add_paragraph(doc, "视频模块使用 OpenCV 的 VideoCapture 读取 MP4、AVI、MOV 等常见格式，并记录路径、分辨率、帧率、总帧数和时长。对于本次最终演示视频，分辨率为 720×720，帧率为 24 fps，共 144 帧；重建不再从普通 RGB 画面中猜测深度，而是使用生成视频时同步保存的深度图、相机内参和相机位姿进行 RGB-D 融合。")
    add_table(doc, ["项目", "数值"], [
        ("演示视频", "data/input_videos/s3dis_area6_office22.mp4"),
        ("帧数", "144"),
        ("帧率", "24 fps"),
        ("分辨率", "720×720"),
        ("重建输入", "144组 color/depth/pose RGB-D 帧"),
    ], widths=[4.0, 9.0])

    add_heading(doc, "2.2 视频到三维模型生成", 2)
    add_paragraph(doc, "针对一般单目视频缺少精确相机位姿和深度的问题，本文将最终演示的重建流程改为 S3DIS 室内场景 RGB-D TSDF 融合。生成脚本从 data/datasets/S3DIS/processed 中的 S3DIS .npy 点云读取 XYZ、RGB 和语义标签，规划一条从场景一侧推进到另一侧的平滑相机路径，并逐帧输出彩色图、深度图和 camera-to-world 位姿。GUI 重建模块读取同名 RGB-D 工程后，将每帧 RGB-D 图像按内参和外参积分进 Open3D ScalableTSDFVolume，最后提取三角网格。")
    add_paragraph(doc, "该方法的优点是几何结构稳定、可复现、适合室内多物体场景，并且输出文件可直接用 MeshLab、Blender 或系统自身导入模块打开。实验场景包含 ceiling、wall、beam、column、window、table、chair、bookcase、board 等多类物体；TSDF 融合使用 144 帧 RGB-D 数据，最终网格包含 191658 个顶点和 303720 个三角面。")
    add_image(doc, "reconstruction_preview.png", 9.2, "图2 基于 S3DIS RGB-D TSDF 融合得到的三维重建结果")

    add_heading(doc, "2.3 三维理解算法", 2)
    add_paragraph(doc, "实例识别模块以点云作为输入，先对点云进行中心化、尺度归一化和定量采样，再利用 PointNet++ 风格的一维卷积特征提取器生成全局和局部特征，最后输出候选中心、尺寸、置信度和类别概率。实例分割模块在点特征上进行语义分类和偏移预测，并结合 DBSCAN 聚类得到独立实例。对于课程演示中的三个主要物体，系统可根据空间聚类把中心球体、左侧柱体、右侧盒体及地面区域区分开。")
    add_image(doc, "segmentation_preview.png", 9.2, "图3 三维实例分割结果预览")

    add_heading(doc, "2.4 材质与光照编辑", 2)
    add_paragraph(doc, "材质编辑模块定义 MaterialProperties 数据结构，包含颜色、粗糙度、金属度、透明度和折射率等参数。系统预置金属、木质、塑料、玻璃、石材、陶瓷、织物七类材质，满足课程要求的五类以上材质替换。光照模块以 Light 为基类，派生环境光、方向光、点光源和聚光灯，支持添加、删除、更新颜色、强度、位置和方向。")
    add_image(doc, "material_preview.png", 14.8, "图4 七类材质替换效果")
    add_image(doc, "lighting_preview.png", 14.0, "图5 不同光照设置效果")

    add_heading(doc, "3 自主训练过程", 1)
    add_paragraph(doc, "课程要求任选一个模块必须包含完全自主训练过程，不能全程直接调用预训练模型。本文选择三维分割模块进行自主训练。训练数据位于 data/datasets/S3DIS/processed，按 train、val、test 划分，样本以 .npy 文件保存，每个样本包含点坐标、颜色/特征和点级标签。训练脚本 training/train_segmentation.py 定义 SegmentationDataset 与 SegmentationTrainer，模型结构位于 models/segmentation_model.py。")
    add_paragraph(doc, "训练模型采用 PointNet++ 风格的层次特征抽象结构：先用最远点采样和球查询获得局部邻域，再通过共享 MLP 提取局部特征，随后用特征传播和一维卷积分类器输出每个点的类别。损失函数为交叉熵损失，优化器为 Adam，学习率初始值为 0.001，并使用 StepLR 衰减。")
    add_table(doc, ["训练项目", "设置"], [
        ("训练对象", "三维点云语义/实例分割模块"),
        ("数据集", "整理后的 S3DIS 室内点云数据集，类别采用 S3DIS 13类标注"),
        ("输入点数", "2048点/样本"),
        ("输入通道", "6维，包含XYZ和颜色/附加特征"),
        ("优化器", "Adam"),
        ("损失函数", "CrossEntropyLoss"),
        ("保存检查点", "data/checkpoints/segmentation_model_final.pth"),
        ("最终记录", "epoch 10，训练准确率约 52.73%，训练损失约 1.4669"),
    ], widths=[4.0, 9.0])
    add_paragraph(doc, "由于本课程项目重点是系统功能贯通而不是单一分割模型刷榜，训练轮数控制在可接受时间内。检查点能够证明模型不是直接全程调用预训练权重，而是通过本地训练脚本产生，并被 GUI 在启动时加载。后续若继续提高分割精度，可增加训练轮数、加入验证集 mIoU 指标、改进数据增强和类别不均衡处理。")

    add_heading(doc, "4 系统实现", 1)
    add_paragraph(doc, "系统入口为 main.py，主窗口由 gui/main_window.py 构建。用户可通过菜单或右侧控制面板依次执行 Open Video、3D Reconstruction、Export Model、Import Model、Instance Detection、Instance Segmentation、Material Editing 和 Lighting Control。三维模型 I/O 由 ModelIO 统一封装，PLY 点云、OBJ/STL/GLB 网格均可被系统读取或导出。")
    add_table(doc, ["文件/目录", "作用"], [
        ("main.py", "程序入口，创建 QApplication 和 MainWindow"),
        ("core/video_processor.py", "视频读取、帧提取、基础预处理"),
        ("tools/generate_s3dis_scene_video.py", "生成 S3DIS 推进视频、RGB-D 序列并完成稳定三维重建"),
        ("core/model_io.py", "OBJ、PLY、STL、GLB 等格式导入导出"),
        ("core/instance_detector.py", "三维实例识别"),
        ("core/instance_segmentor.py", "三维实例分割"),
        ("core/material_editor.py", "七类材质参数与替换"),
        ("core/lighting_system.py", "环境光、方向光、点光源、聚光灯管理"),
        ("data/output_models", "最终重建模型与分割模型输出"),
    ], widths=[5.2, 8.0])
    add_paragraph(doc, "最终交付的关键数据包括：s3dis_area6_office22.mp4 作为系统运行视频输入；s3dis_area6_office22_rgbd 作为配套 RGB-D 工程；s3dis_area6_office22_tsdf_mesh.ply 作为 TSDF 重建模型；s3dis_area6_office22_segmented_labels.ply 和 s3dis_area6_office22_instances 作为语义分割结果；s3dis_area6_office22_summary.json 记录重建统计信息。")

    add_heading(doc, "5 结果分析与结论", 1)
    add_paragraph(doc, "实验结果表明，本文系统能够完成课程要求中的完整端到端流程。视频模块能够读取视频信息并抽取帧；三维生成模块能够基于 S3DIS 推进视频及配套 RGB-D 数据恢复室内场景模型；模型导出文件可以保存为通用 PLY；系统支持再导入外部 STL、PLY 等模型；实例识别和分割模块能够对点云/网格进行物体级分析；材质和光照模块能够实时改变场景外观。")
    add_table(doc, ["评价项", "结果"], [
        ("重建完整性", "得到连续三角网格，191658 个顶点、303720 个三角面"),
        ("导出格式", "PLY 已生成"),
        ("分割结果", "输出 s3dis_area6_office22_segmented_labels.ply 和各语义对象 PLY"),
        ("材质数量", "7类，超过课程要求的5类"),
        ("光照类型", "4类，覆盖环境光、方向光、点光源和聚光灯"),
        ("自主训练", "保存 epoch 10 检查点，训练准确率约52.73%"),
    ], widths=[4.0, 9.0])
    add_paragraph(doc, "系统仍有可改进之处。S3DIS RGB-D TSDF 融合适合带深度和位姿的受控室内数据，对普通真实手持 RGB 视频仍依赖额外的 SLAM、深度估计或传感器数据；当前分割模型训练轮数较少，复杂真实场景下的语义准确率仍需提升；GUI 中实时渲染材质和光照主要体现为颜色与光照参数变化，距离完整 PBR 渲染仍有差距。总体而言，本系统已经形成可运行、可演示、可导出、可训练的课程项目闭环，满足本次计算机视觉课程论文与系统实践要求。")

    add_heading(doc, "致谢", 1)
    add_paragraph(doc, "感谢王海琨老师在计算机视觉课程中对三维重建、目标识别和系统工程实践提出的要求与指导。本文系统实现过程中参考了开源社区关于 OpenCV、PyTorch、Open3D、PointNet++、TSDF 融合与室内三维语义分割的相关资料。")

    add_heading(doc, "参考文献", 1)
    refs = [
        "[1] Curless B, Levoy M. A Volumetric Method for Building Complex Models from Range Images. SIGGRAPH, 1996.",
        "[2] Armeni I, Sener O, Zamir A R, et al. 3D Semantic Parsing of Large-Scale Indoor Spaces. CVPR, 2016.",
        "[3] Qi C R, Yi L, Su H, et al. PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space. NeurIPS, 2017.",
        "[4] Qi C R, Litany O, He K, et al. Deep Hough Voting for 3D Object Detection in Point Clouds. ICCV, 2019.",
        "[5] Jiang L, Zhao H, Shi S, et al. PointGroup: Dual-Set Point Grouping for 3D Instance Segmentation. CVPR, 2020.",
        "[6] Newcombe R A, Izadi S, Hilliges O, et al. KinectFusion: Real-Time Dense Surface Mapping and Tracking. ISMAR, 2011.",
        "[7] Zhou Q Y, Park J, Koltun V. Open3D: A Modern Library for 3D Data Processing. arXiv:1801.09847, 2018.",
        "[8] Bradski G. The OpenCV Library. Dr. Dobb's Journal of Software Tools, 2000.",
    ]
    for ref in refs:
        add_paragraph(doc, ref, first_line=False, spacing_after=2)


def main():
    doc = Document()
    cover(doc)
    abstract_pages(doc)
    toc(doc)
    body(doc)
    doc.save(OUT)
    print(OUT)


if __name__ == "__main__":
    main()
