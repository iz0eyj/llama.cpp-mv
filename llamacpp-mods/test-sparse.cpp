#include "llama.h"
#include "common.h"
#include <cstdio>
#include <vector>
#include <cstring>

int main(int argc, char ** argv) {
    if (argc < 2) return 1;
    const char * model_path = argv[1];

    common_params params;
    params.model.path = model_path;
    params.embedding = true;
    params.pooling = LLAMA_POOLING_TYPE_CLS;
    params.n_ctx   = 512;
    params.n_batch = 512;
    params.n_ubatch = 512;
    params.cpuparams.n_threads = 4;

    common_init();

    printf("init model...\n"); fflush(stdout);
    common_init_result result = common_init_from_params(params);
    if (!result.model || !result.context) {
        fprintf(stderr, "init failed\n");
        return 1;
    }
    printf("model loaded\n"); fflush(stdout);

    llama_model * model = result.model;
    llama_context * ctx = result.context;
    const llama_vocab * vocab = llama_model_get_vocab(model);

    const char * prompt = "Samarcanda";
    int n = llama_tokenize(vocab, prompt, (int)strlen(prompt), nullptr, 512, true, true);
    std::vector<llama_token> toks(n);
    llama_tokenize(vocab, prompt, (int)strlen(prompt), toks.data(), n, true, true);

    llama_batch batch = llama_batch_init(n, 0, 1);
    for (int i = 0; i < n; i++) {
        batch.token[i] = toks[i]; batch.pos[i] = i;
        batch.n_seq_id[i] = 1; batch.seq_id[i][0] = 0;
        batch.logits[i] = true;
    }
    batch.n_tokens = n;

    printf("decode...\n"); fflush(stdout);
    int r = llama_decode(ctx, batch);
    printf("decode=%d\n", r); fflush(stdout);

    if (r == 0) {
        printf("--- DENSE ---\n");
        float * d = llama_get_embeddings_seq(ctx, 0);
        if (d) printf("dense[0]=%f ... [1023]=%f\n", d[0], d[1023]);

        printf("--- SPARSE ---\n");
        for (int i = 0; i < n; i++) {
            float w = llama_get_embeddings_sparse_ith(ctx, i);
            printf("  token[%d] id=%d sparse=%.6f\n", i, toks[i], w);
        }
    }

    llama_batch_free(batch);
    common_free();
    printf("done\n");
    return 0;
}
