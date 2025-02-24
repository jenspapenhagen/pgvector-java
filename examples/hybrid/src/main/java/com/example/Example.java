package com.example;

import com.pgvector.PGvector;
import io.github.ollama4j.OllamaAPI;
import io.github.ollama4j.exceptions.OllamaBaseException;
import io.github.ollama4j.models.embeddings.OllamaEmbedRequestModel;
import io.github.ollama4j.models.embeddings.OllamaEmbedResponseModel;

import java.io.IOException;
import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.sql.Statement;
import java.util.List;

public class Example {
    public static void main(String[] args) throws IOException, SQLException {        
        //DB: connect to the db
        final Connection connection = DriverManager.getConnection("jdbc:postgresql://localhost:5432/pgvector_example");

        //DB: load extension and drop the table
        final Statement setupStatement = connection.createStatement();
        setupStatement.executeUpdate("CREATE EXTENSION IF NOT EXISTS vector");
        setupStatement.executeUpdate("DROP TABLE IF EXISTS documents");

        //DB: add the extension to the PostgreSQL
        PGvector.addVectorType(connection);

        //DB: create the table
        //HINT: change the embedding size on other models

        //Example:
        //'nomic-embed-text' # 768 dimensions
        //'mxbai-embed-large' # 1024 dimensions
        //
        // or use the Embedding Leaderboard from Hugginface (MTEB)
        // https://huggingface.co/spaces/mteb/leaderboard

        // WARNING
        // the primary vector limitation is due to the default size of the PostgreSQL page (8KB),
        // which is not adjustable.
        // that mines for foating point 32 is the vector limit 2048
        // 8 KB / 4 B (fp32) = vector limit
        // calucation: 8192 / 4 = 2048

        // What can be done, about it?
        // use halfvec for fp16 (max. 4092)
        // use sparsevec for int8 (max. 8192)
        // use bit for binary (max. 65536)
        // read more: https://www.mixedbread.com/blog/binary-mrl
        final Statement createTableStatement = connection.createStatement();
        createTableStatement.executeUpdate("CREATE TABLE documents (id bigserial PRIMARY KEY, content text, embedding vector(384))");

        //DB: create index
        // for a more practical/realistically usecase. We create an index of document outside the table, instead of inside the table.
        // Because we want blocking other SQL action on the table.
        // more infos: https://github.com/pgvector/pgvector?tab=readme-ov-file#hnsw
        //final Statement createIndexStatement = connection.createStatement();
        // halfvec
        // createIndexStatement.executeUpdate("CREATE INDEX ON documents USING hnsw ((embedding::halfvec(3072)) halfvec_cosine_ops);"
        // binary
        //createIndexStatement.executeUpdate("CREATE INDEX ON documents USING hnsw ((binary_quantize(embedding)::bit(3072)) bit_hamming_ops);"

        //OLLAMA: use a remote source with heavier GPU hardware
        // why:
        // network latency/delay < embedding speed with better GPU
        final String host = "http://localhost:11434/";
        final OllamaAPI ollamaAPI = new OllamaAPI(host);

        //example data a simple strings
        final List<String> input = List.of(
                "The dog is barking",
                "The cat is purring",
                "The bear is growling");


        //more info to the ollama4j api: https://ollama4j.github.io/ollama4j/apis-generate/generate-embeddings/
        final OllamaEmbedResponseModel embeddings = ollamaAPI.embed(new OllamaEmbedRequestModel("albertogg/multi-qa-minilm-l6-cos-v1", input));

        //DB: fill table
        for (int i = 0; i < input.size(); i++) {
            PreparedStatement insertStmt = connection.prepareStatement("INSERT INTO documents (content, embedding) VALUES (?, ?)");
            insertStmt.setString(1, input.get(i));
            insertStmt.setObject(2, new PGvector(embeddings.getEmbeddings().get(i)));
            insertStmt.executeUpdate();
        }

        //Search: search with exmple query
        final String query = "growling bear";
        final List<Double> queryEmbedding = ollamaAPI.embed(new OllamaEmbedRequestModel("albertogg/multi-qa-minilm-l6-cos-v1", List.of(query))).getEmbeddings().getFirst();

        //Reciprocal Rank Fusion
        //        RRF(d) = Σ(r ∈ R) 1 / (k + r(d))
        //
        //        Where:
        //        - d is a document
        //            - R is the set of rankers (retrievers)
        //            - k is a constant (typically 60)
        //            - r(d) is the rank of document d in ranker r
        //why:
        //Reciprocal Ranking - It gives more weight to higher ranks (lower rank numbers).
        //                     This ensures that documents ranked highly by multiple retrievers are
        //                     favoured in the final ranking.
        //Diminishing Returns - The contribution to the score decreases non-linearly as rank increases.
        //                      This model shows the intuition that the difference in relevance between
        //                      ranks 1 and 2 is likely larger than between ranks 100 and 101.
        //Rank Aggregation - It effectively combines evidence from multiple sources.
        //                   This makes the final ranking more robust and less susceptible
        //                   to the quirks or biases of any single retriever.
        //Normalization - The constant k acts as a smoothing factor.
        //                It prevents any single retriever from dominating the results.
        //more info:
        // - https://medium.com/@devalshah1619/mathematical-intuition-behind-reciprocal-rank-fusion-rrf-explained-in-2-mins-002df0cc5e2a
        //paper:
        // - https://doi.org/10.1145/1571941.157211
        // - https://doi.org/10.1145/3166072.3166084
        // - https://doi.org/10.48550/arXiv.2108.06130
        // - https://doi.org/10.1145/3308774.3308781
        // - https://doi.org/10.1109/ACCESS.2023.3295776

        final double k = 60;
        //why 60?
        //Empirical Performance - It performs well across various datasets and retrieval tasks
        //Balancing Influence - It provides a good balance between the influence of top-ranked and lower-ranked items
        //Effective Tie-Breaking - It breaks effectively, especially for lower-ranked items
        //                         where small differences in the original rankings might not be significant

        //DB: build the query
        PreparedStatement prepareSearchStatement = connection.prepareStatement(HYBRID_SQL);
        prepareSearchStatement.setObject(1, new PGvector(queryEmbedding));
        prepareSearchStatement.setObject(2, new PGvector(queryEmbedding));
        prepareSearchStatement.setString(3, query);
        prepareSearchStatement.setDouble(4, k);
        prepareSearchStatement.setDouble(5, k);

        //DB: exicute the query
        ResultSet rs = prepareSearchStatement.executeQuery();

        //DB: display the result set
        while (rs.next()) {
            System.out.printf("document: %d, RRF score: %f%n", rs.getLong("id"), rs.getDouble("score"));
        }

        //DB: close the connection
        connection.close();
    }

    public static final String HYBRID_SQL = """
            WITH semantic_search AS (
                SELECT id, RANK () OVER (ORDER BY embedding <=> ?) AS rank
                FROM documents
                ORDER BY embedding <=> ?
                LIMIT 20
            ),
            keyword_search AS (
                SELECT id, RANK () OVER (ORDER BY ts_rank_cd(to_tsvector('english', content), query) DESC)
                FROM documents, plainto_tsquery('english', ?) query
                WHERE to_tsvector('english', content) @@ query
                ORDER BY ts_rank_cd(to_tsvector('english', content), query) DESC
                LIMIT 20
            )
            SELECT
                COALESCE(semantic_search.id, keyword_search.id) AS id,
                COALESCE(1.0 / (? + semantic_search.rank), 0.0) +
                COALESCE(1.0 / (? + keyword_search.rank), 0.0) AS score
            FROM semantic_search
            FULL OUTER JOIN keyword_search ON semantic_search.id = keyword_search.id
            ORDER BY score DESC
            LIMIT 5
            """;
}
