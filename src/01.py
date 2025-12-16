from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.runnables import RunnableConfig

from pydantic import BaseModel, Field
import operator
from typing import Annotated, Optional, Literal, List, Callable, Any  # 导入 List 以增强兼容性

from dotenv import load_dotenv
load_dotenv()

# --- 1. Pydantic 结构定义 ---


class Task(BaseModel):
    """一个包含自引用子任务的层级任务结构。"""

    # 基础属性，使用原生类型 list 或 str
    name: str = Field(description="任务名称")
    content: str = Field(description="任务内容描述")
    status: Literal["pending", "processing", "completed"] = Field(
        description="任务状态", default="pending")

    # 核心层级属性：自引用类型
    # V2 标准：直接使用字符串 'Task' 来引用自身，无需从 typing 导入 ForwardRef
    sub_tasks: Optional[list['Task']] = Field(
        description="当前任务的子任务列表",
        default_factory=list
    )

    # Pydantic V2 继承 BaseModel 后，会自动处理字符串形式的前向引用，
    # 理论上不再需要 Task.model_rebuild()，但如果遇到问题，可以手动调用。


def traverse_tasks(tasks: List[Task], level: int = 0, processor: Optional[Callable[[Task, int], Any]] = None):
    """
    递归遍历任务列表及其所有子任务。

    Args:
        tasks (List[Task]): 当前层级的任务列表。
        level (int): 当前任务的层级深度 (0 表示顶层)。
        processor (Callable): 对每个任务执行的处理函数，接收 Task 和 level 作为参数。
    """
    indent = "    " * level  # 用于打印缩进

    for task in tasks:
        # 1. 对当前任务进行处理（打印、修改、记录等）
        if processor:
            processor(task, level)
        else:
            # 默认处理：打印信息
            print(f"{indent}[{level}] {task.name} ({task.status})")

        # 2. 递归调用：如果存在子任务，则对子任务列表进行递归
        if task.sub_tasks:
            traverse_tasks(task.sub_tasks, level + 1, processor)


class ProgramState(BaseModel):
    """LangGraph 工作流的全局状态。"""
    # message
    message: str = Field(
        description="用户输入", default="unknow")

    # 环节
    phase: Literal["Analysis", "Design", "Coding", "unknow"] = Field(
        description="当前环节", default="unknow")

    # Analysis
    analysis_task: Task = Field(description="需求分析任务", default=None)
    # Design
    design_task: Task = Field(description="项目设计任务", default=None)
    # Coding
    coding_task: Task = Field(description="编码任务", default=None)
    # Testing
    # Deployment


# --- 2. 节点函数定义 ---

# 建议节点返回一个字典，LangGraph 会将其与旧状态合并
def analysis_node(state: ProgramState) -> ProgramState:
    print("--- 1. 进入分析节点 (Analysis Node) ---")

    # 构建提示词
    analysis_prompt = f"""
        你是一位高级需求分析师。请对用户提出的需求进行详细分析，将其拆解具体的、可操作的、有意义的子任务。
        任务可以嵌套,参考Task类的定义
        用户需求：{state.message}
    """

    # 使用 TaskList 封装模型配置结构化输出
    analysis_model = model.with_structured_output(Task)
    # 调用 LLM，强制其输出 TaskList 结构
    try:
        analysis_task: Task = analysis_model.invoke(
            analysis_prompt, config)
        print("=======================")
        print(analysis_task)
        print("=======================")
    except Exception as e:
        print(f"分析节点LLM调用失败: {e}")
        analysis_task = Task(name="Analysis Failed",
                             content=str(e), status="completed")

    # print(f"✅ 分析完成，生成了 {len(analysis_tasks)} 个任务。")

    # 返回字典，更新状态
    return {
        "analysis_task": analysis_task,
        "phase": "Analysis"
    }


def design_node(state: ProgramState) -> ProgramState:
    print("\n--- 2. 进入设计节点 (Design Node) ---")

    # 示例逻辑：打印分析结果并占位
    print(f"接收到 {len(state.analysis_task.sub_tasks)} 个分析任务，准备进行设计。")

    # 这里应该添加 LLM 逻辑来生成设计任务，这里仅为占位
    design_task = Task(name="Define Architecture",
                       content="确定项目技术栈和架构。", status="pending")

    return {
        "design_task": design_task,
        "phase": "Design"
    }


def coding_node(state: ProgramState) -> ProgramState:
    print("\n--- 3. 进入编码节点 (Coding Node) ---")

    # 示例逻辑：打印设计结果并占位
    print(f"接收到 {len(state.design_task.sub_tasks)} 个设计任务，准备开始编码。")

    # 这里应该添加 LLM 逻辑来生成代码或编码任务
    coding_task = Task(name="Setup Project",
                       content="初始化项目骨架。", status="pending")

    return {
        "coding_task": coding_task,
        "phase": "Coding"
    }


# --- 3. 初始化与执行 ---

# 初始化模型
# 请确保环境变量 (如 DEEPSEEK_API_KEY) 已在 .env 文件中设置
print("⏳ 正在初始化模型...")
model = init_chat_model("deepseek-chat")
print("✅ 模型初始化完成。")

# 构建工作流
workflow = StateGraph(ProgramState)

# 添加节点
workflow.add_node("analysis_node", analysis_node)
workflow.add_node("design_node", design_node)
workflow.add_node("coding_node", coding_node)

# 添加边连接节点
workflow.add_edge(START, "analysis_node")
workflow.add_edge("analysis_node", "design_node")
workflow.add_edge("design_node", "coding_node")
workflow.add_edge("coding_node", END)

# 编译代理
checkpointer = InMemorySaver()
graph = workflow.compile(checkpointer=checkpointer)

# 配置与运行
config: RunnableConfig = {"configurable": {"thread_id": "1"}}
user_input = "使用Python完成一个极简的web项目，用户可以发布和查看文章。"
initial_state = {"message": user_input}

print(f"\n==========================================")
print(f"🚀 开始执行工作流，用户需求: {user_input}")
print(f"==========================================")

# 调用工作流
response_dict = graph.invoke(initial_state, config)
response: ProgramState = ProgramState.model_validate(response_dict)
print("\n==========================================")
print("✨ 工作流执行结果 (最终状态):")
print("==========================================")
print(f"最终环节: {response.phase}")
print("--- 需求分析任务 (Analysis Tasks) ---")
for i, task in enumerate(response.analysis_task.sub_tasks):
    print(f"  {i+1}. {task.name} ({task.status})")
    print(f"  \t{task.content}")
print("--- 项目设计任务 (Design Tasks) ---")
for i, task in enumerate(response.design_task.sub_tasks):
    print(f"  {i+1}. {task.name} ({task.status})")
    print(f"  \t{task.content}")
print("--- 编码任务 (Coding Tasks) ---")
for i, task in enumerate(response.coding_task.sub_tasks):
    print(f"  {i+1}. {task.name} ({task.status})")
    print(f"  \t{task.content}")
