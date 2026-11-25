const STEPS = [
  { label: "取运行时", detail: "每个栈一次" },
  { label: "构建镜像", detail: "已有则跳过" },
  { label: "装配模型包", detail: "绑参数" },
  { label: "部署", detail: "本机容器" },
  { label: "调用", detail: "直连宿主机端口" },
];

export function FlowSteps({ active }: { active: number }) {
  return (
    <ol className="steps">
      {STEPS.map((step, index) => {
        const state = index < active ? "done" : index === active ? "on" : "";
        return (
          <li key={step.label} className={`step ${state}`.trim()}>
            <strong>
              {index + 1}. {step.label}
            </strong>
            <span>{step.detail}</span>
          </li>
        );
      })}
    </ol>
  );
}
