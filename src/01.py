from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.runnables import RunnableConfig

from pydantic import BaseModel, Field
import operator
from typing import Annotated, Literal, List  # 导入 List 以增强兼容性

from dotenv import load_dotenv
load_dotenv()

# --- 1. Pydantic 结构定义 ---


class Task(BaseModel):
    """定义工作流中的一个基本任务单元。"""
    name: str = Field(description="任务名称", default="未知任务")
    content: str = Field(description="任务内容描述", default="未知内容")
    # 修复：默认值设为允许值列表中的一个
    status: Literal["pending", "processing", "completed"] = Field(
        description="任务状态", default="pending")


class TaskList(BaseModel):
    """用于封装和解析 Task 列表的容器模型，适配 LangChain Structured Output 要求。"""
    analysis_tasks: List[Task] = Field(
        description="需求分析后拆解出的所有任务列表。",
    )


class ProgramState(BaseModel):
    """LangGraph 工作流的全局状态。"""
    # message
    message: Annotated[List[str], operator.add] = Field(
        description="用户输入", default=list)

    # 环节
    phase: Literal["Analysis", "Design", "Coding", "unknow"] = Field(
        description="当前环节", default="unknow")

    # Analysis
    analysis_tasks: List[Task] = Field(
        description="需求分析任务列表", default_factory=list)
    # Design
    design_tasks: List[Task] = Field(
        description="项目设计任务列表", default_factory=list)
    # Coding
    coding_tasks: List[Task] = Field(
        description="编码任务列表", default_factory=list)
    # Testing
    # Deployment


# --- 2. 节点函数定义 ---

# 建议节点返回一个字典，LangGraph 会将其与旧状态合并
def analysis_node(state: ProgramState) -> ProgramState:
    print("--- 1. 进入分析节点 (Analysis Node) ---")

    # 构建提示词
    analysis_prompt = f"""
        你是一位高级需求分析师。请对用户提出的需求进行详细分析，将其拆解为 3 到 5 个具体的、可操作的、有意义的子任务。
        
        用户需求：{state.message}
        
        请严格按照 JSON 格式输出，任务列表必须包含在 'analysis_tasks' 键下，并符合 Task 结构。
    """

    # 使用 TaskList 封装模型配置结构化输出
    structured_model = model.with_structured_output(TaskList)

    # 调用 LLM，强制其输出 TaskList 结构
    try:
        response_container: TaskList = structured_model.invoke(
            analysis_prompt, config)
        tasks_list = response_container.analysis_tasks
    except Exception as e:
        print(f"分析节点LLM调用失败: {e}")
        tasks_list = [Task(name="Analysis Failed",
                           content=str(e), status="completed")]

    print(f"✅ 分析完成，生成了 {len(tasks_list)} 个任务。")

    # 返回字典，更新状态
    return {
        "analysis_tasks": tasks_list,
        "phase": "Analysis"
    }


def design_node(state: ProgramState) -> ProgramState:
    print("\n--- 2. 进入设计节点 (Design Node) ---")

    # 示例逻辑：打印分析结果并占位
    print(f"接收到 {len(state.analysis_tasks)} 个分析任务，准备进行设计。")

    # 这里应该添加 LLM 逻辑来生成设计任务，这里仅为占位
    design_tasks = [
        Task(name="Define Architecture", content="确定项目技术栈和架构。", status="pending"),
        Task(name="Database Schema", content="设计数据库表结构。", status="pending")
    ]

    return {
        "design_tasks": design_tasks,
        "phase": "Design"
    }


def coding_node(state: ProgramState) -> ProgramState:
    print("\n--- 3. 进入编码节点 (Coding Node) ---")

    # 示例逻辑：打印设计结果并占位
    print(f"接收到 {len(state.design_tasks)} 个设计任务，准备开始编码。")

    # 这里应该添加 LLM 逻辑来生成代码或编码任务
    coding_tasks = [
        Task(name="Setup Project", content="初始化项目骨架。", status="pending"),
        Task(name="Implement Endpoints",
             content="实现 Web API 接口。", status="pending")
    ]

    return {
        "coding_tasks": coding_tasks,
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
for i, task in enumerate(response.analysis_tasks):
    print(f"  {i+1}. {task.name} ({task.status})")
    print(f"  \t{task.content}")
print("--- 项目设计任务 (Design Tasks) ---")
for i, task in enumerate(response.design_tasks):
    print(f"  {i+1}. {task.name} ({task.status})")
    print(f"  \t{task.content}")
print("--- 编码任务 (Coding Tasks) ---")
for i, task in enumerate(response.coding_tasks):
    print(f"  {i+1}. {task.name} ({task.status})")
    print(f"  \t{task.content}")
