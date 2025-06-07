import pickle
import random
from sentence_transformers import SentenceTransformer

from data.conceptnet_graphbuilder import ConceptNetGraphBuilder
from data.commonsenseqa import CommonsenseQA
from env_rl import KGEnvRL
from gpt import simulate_fn_gpt
from policy import PolicyNet, TriplePolicyNet
from prompt_construction import PromptConstructorMAB
from prompt_templates import TEMPLATE_FUNCS
from train_mab import train_mab_on_data
from train_rl import train_rl_multi_epoch
from utils import load_or_cache_conceptnet
from configs.config import (
    QA_PATH, CONCEPTNET_PATH,
    PKL_CCN_PATH, ENTITY_TO_VEC_PATH,
    TRIPLE_TO_VEC_PATH, EXTRACTED_TRAIN_PATH,
    USE_TRIPLE_POLICY, EMBEDDING_DIM,
    HIDDEN_DIM, NUM_EPOCHS, sample_size,
    REPEATS_PER_SAMPLE, MODEL_NAME
)

def step1_load_conceptnet():
    """
    Step 1: Load ConceptNet triples from raw CSV or cache.
    """
    cache_path = PKL_CCN_PATH(sample_size)
    df = load_or_cache_conceptnet(CONCEPTNET_PATH, sample_size, cache_path)
    print(f"✅ Loaded {len(df)} edges")
    return df

def step2_build_graph(df):
    """
    Step 2: Build the Knowledge Graph from ConceptNet triples.
    """
    builder = ConceptNetGraphBuilder(df)
    KG = builder.build_graph()
    print(f"✅ Graph built: {KG.number_of_nodes()} nodes, {KG.number_of_edges()} edges")
    return KG, builder

def step3_build_embeddings(builder):
    """
    Step 3: Build and save entity and triple embeddings using SentenceTransformer.
    """
    entity_vecs = builder.build_entity_embeddings()
    triple_vecs = builder.build_triple_embeddings(use_special_tokens=True)
    with open(ENTITY_TO_VEC_PATH, "wb") as f:
        pickle.dump(entity_vecs, f)
    with open(TRIPLE_TO_VEC_PATH, "wb") as f:
        pickle.dump(triple_vecs, f)
    return entity_vecs, triple_vecs

def step4_load_dataset():
    """
    Step 4: Load CommonsenseQA dataset and return QA samples.
    """
    csqa = CommonsenseQA(QA_PATH)
    samples = csqa.get_all()
    return samples

def step5_train_rl(KG, samples, entity_to_vec, triplet_to_vec):
    """
    Step 5: Train RL agent to extract reasoning paths on KG.
    """
    model = SentenceTransformer(MODEL_NAME)
    env = KGEnvRL(
        graph=KG,
        samples=samples,
        entity_to_vec=entity_to_vec,
        triplet_to_vec=triplet_to_vec,
        embed_model=model,
        max_steps=10,
        lambda_context=0.7,
        lambda_concise=0.3,
        use_triple_embedding=USE_TRIPLE_POLICY
    )

    if USE_TRIPLE_POLICY:
        policy_net = TriplePolicyNet(triple_dim=EMBEDDING_DIM, hidden_dim=HIDDEN_DIM)
        results = train_rl_multi_epoch(
            policy_net=policy_net,
            env=env,
            samples=samples,
            num_epochs=NUM_EPOCHS,
            repeats_per_sample=REPEATS_PER_SAMPLE,
            use_triple=True,
            triplet_to_vec=triplet_to_vec
        )
    else:
        policy_net = PolicyNet(input_dim=EMBEDDING_DIM * 3, hidden_dim=HIDDEN_DIM)
        results = train_rl_multi_epoch(
            policy_net=policy_net,
            env=env,
            samples=samples,
            num_epochs=NUM_EPOCHS,
            repeats_per_sample=REPEATS_PER_SAMPLE,
            use_triple=False,
            entity_to_vec=entity_to_vec
        )

    with open(EXTRACTED_TRAIN_PATH, "wb") as f:
        pickle.dump(results, f)
    print(f"✅ Saved {len(results)} successful reasoning paths to '{EXTRACTED_TRAIN_PATH}'")
    return results

def step6_prepare_for_mab(results, samples):
    """
    Step 6: Attach QA ground truth (correct answer, options) to RL results.
    """
    qid2sample = {s['question_id']: s for s in samples}
    for r in results:
        if 'correct_answer' not in r and r['question_id'] in qid2sample:
            s = qid2sample[r['question_id']]
            r['correct_answer'] = s['correct_answer']
            r['options'] = s['options']
    return results

def step7_train_mab(results):
    """
    Step 7: Train Multi-Armed Bandit (MAB) to select best prompting strategy.
    """
    mab = PromptConstructorMAB(TEMPLATE_FUNCS)
    acc, log = train_mab_on_data(results, mab, simulate_fn_gpt, max_samples=100, top_k_paths=3)
    print(f"🎯 Accuracy on train set: {acc:.2%}")
    return acc, log

def main():
    """
    Main function to run the full pipeline from KG construction to GPT prompting.
    """
    df = step1_load_conceptnet()
    KG, builder = step2_build_graph(df)
    entity_vecs, triple_vecs = step3_build_embeddings(builder)
    samples = step4_load_dataset()

    # Select embedding type
    if USE_TRIPLE_POLICY:
        entity_to_vec = None
        triplet_to_vec = triple_vecs
    else:
        entity_to_vec = entity_vecs
        triplet_to_vec = None

    results = step5_train_rl(KG, samples, entity_to_vec, triplet_to_vec)

    with open(EXTRACTED_TRAIN_PATH, "rb") as f:
        results = pickle.load(f)

    results = step6_prepare_for_mab(results, samples)
    step7_train_mab(results)

if __name__ == '__main__':
    main()
