from metagpt.aflow.scripts.evaluator import Evaluator


class EvaluationUtils:
    def __init__(self, root_path: str):
        self.root_path = root_path

    async def evaluate_initial_round(self, optimizer, graph_path, directory, validation_n, data):
        # 使用 optimizer 的 graph_utils 来加载图
        optimizer.graph = optimizer.graph_utils.load_graph(optimizer.round, graph_path)
        evaluator = Evaluator(eval_path=directory)

        for i in range(validation_n):
            score, avg_cost, total_cost = await evaluator.graph_evaluate(
                optimizer.dataset,
                optimizer.graph,
                {"dataset": optimizer.dataset, "llm_config": optimizer.execute_llm_config},
                directory,
                is_test=False,
            )

            new_data = optimizer.data_utils.create_result_data(optimizer.round, score, avg_cost, total_cost)
            data.append(new_data)

            result_path = optimizer.data_utils.get_results_file_path(graph_path)
            optimizer.data_utils.save_results(result_path, data)

        return data

    async def evaluate_graph(self, optimizer, directory, validation_n, data, initial=False, va_list=None, 
                             gnn_optimize=False, model_name=None, ckp=None, memory=None, cost = None):
        evaluator = Evaluator(eval_path=directory)
        sum_score = 0

        for i in range(validation_n):
            score, avg_cost, total_cost = await evaluator.graph_evaluate(
                optimizer.dataset,
                optimizer.graph,
                {"dataset": optimizer.dataset, "llm_config": optimizer.execute_llm_config},
                directory,
                is_test=False,
                va_list=va_list,
                gnn_optimize=gnn_optimize,
                model_name=model_name,
                ckp=ckp,
                memory=memory,
            )
            total_cost += cost if cost is not None else 0
            cur_round = optimizer.round + 1 if initial is False else optimizer.round

            new_data = optimizer.data_utils.create_result_data(cur_round, score, avg_cost, total_cost)
            data.append(new_data)

            result_path = optimizer.data_utils.get_results_file_path(f"{optimizer.root_path}/workflows")
            optimizer.data_utils.save_results(result_path, data)

            sum_score += score

        return sum_score / validation_n

    async def evaluate_graph_test(self, optimizer, directory, is_test=True, dataset_file=None):
        evaluator = Evaluator(eval_path=directory)
        return await evaluator.graph_evaluate(
            optimizer.dataset,
            optimizer.graph,
            {"dataset": optimizer.dataset, "llm_config": optimizer.execute_llm_config},
            directory,
            is_test=is_test,
            dataset_file=dataset_file,
        )

    async def load_dataset_embedding(self, dataset):
        evaluator = Evaluator(eval_path=None)
        return await evaluator.load_dataset_embedding(dataset)