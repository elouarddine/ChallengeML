import time
import warnings
warnings.filterwarnings("ignore")  # <= UNE SEULE INSTRUCTION : coupe tous les warnings

from src import Preprocess, Models, Evaluate

_state = {"pre": None, "models": None}

def _step(text, sleep_s=0.15):
    print("AutoML | " + text, flush=True)
    time.sleep(sleep_s)

def _print_report(rep):
    if not isinstance(rep, dict): 
        return
    try:
        from IPython.display import clear_output
        clear_output(wait=True)
    except Exception:
        print("\033[2J\033[H", end="", flush=True)

    print("\n========== AutoML | MODEL SELECTION REPORT ==========", flush=True)
    print("Task type      : " + str(rep.get("task_type")), flush=True)
    print("Imbalanced     : " + ("YES" if rep.get("is_imbalanced") else "NO"), flush=True)
    print("Main metric    : " + str(rep.get("main_metric")), flush=True)

    print("\n--- Light optimisation: tested models (main metric) ---", flush=True)
    tested_light = rep.get("tested_light", [])
    for d in tested_light:
        print("  - " + str(d.get("name")) + " => " + str(d.get("score")), flush=True)

    print("\n--- Selected Top-3 after Light ---", flush=True)
    top_light = rep.get("top_light", [])
    for d in top_light:
        print("  * " + str(d.get("name")) + " => " + str(d.get("score")), flush=True)

    print("\n--- Full optimisation sur les modeles selectionné (main metric) ---", flush=True)
    tested_full = rep.get("tested_full", [])
    if len(tested_full) == 0:
        print("  (No Full stage, kept Light best)", flush=True)
    else:
        for d in tested_full:
            print("  - " + str(d.get("name")) + " => " + str(d.get("score")), flush=True)

    print("\nFINAL BEST MODEL : " + str(rep.get("best_model")) + "  (stage=" + str(rep.get("best_stage")) + ")", flush=True)
    
    print("\n------- Validation metrics du model selectionné sur DEV -------", flush=True)
    val_scores = rep.get("val_scores")
    if isinstance(val_scores, dict) and len(val_scores) > 0:
        for k, v in val_scores.items():
            print("  - " + str(k) + " => " + str(v), flush=True)
    else:
        print("  (No validation split / no val scores)", flush=True)

    print("=====================================================\n", flush=True)

def fit(data_dest):
    _step("Loading dataset (.data/.type/.solution)")
    pre = Preprocess(data_dest)
    pre.load()

    _step("Split train/validation/test")
    pre.split()

    _step("Model selection (Light -> Full + Voting)")
    models = Models(pre)
    models.select_best_model(n_jobs=1)  # n_jobs=1 => moins de spam / warnings

    rep = models.get_selection_report()
    _print_report(rep)

    _step("Saving results (selection)")
    run_path = models.save_results(stage="selection")
    print("AutoML | Resultat d'entraînnement enregistrer dans: " + str(run_path), flush=True)

    _state["pre"] = pre
    _state["models"] = models

    _step("Done")
    return models.best_name

def eval():
    if _state["pre"] is None or _state["models"] is None:
        raise RuntimeError("Call automl.fit(data_dest) before automl.eval().")

    _step("Evaluating best model on test")
    evaluator = Evaluate(preprocessor_obj=_state["pre"], models_obj=_state["models"])
    out = evaluator.evaluate_test()

    if hasattr(evaluator, "save_results"):
        _step("Saving results (test)")
        test_path = evaluator.save_results(stage="test")
        print("AutoML | Test results saved to: " + str(test_path), flush=True)

    _step("Done")
    return out
