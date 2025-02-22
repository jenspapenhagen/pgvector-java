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
    public static void main(String[] args) throws IOException, SQLException {        final Connection conn = DriverManager.getConnection("jdbc:postgresql://localhost:5432/pgvector_example");

        final Statement setupStmt = conn.createStatement();
        setupStmt.executeUpdate("CREATE EXTENSION IF NOT EXISTS vector");
        setupStmt.executeUpdate("DROP TABLE IF EXISTS documents");

        PGvector.addVectorType(conn);

        final Statement createStmt = conn.createStatement();
        createStmt.executeUpdate("CREATE TABLE documents (id bigserial PRIMARY KEY, content text, embedding vector(384))");

        final String host = "http://localhost:11434/";
        final OllamaAPI ollamaAPI = new OllamaAPI(host);

        final List<String> input = List.of(
                "The dog is barking",
                "The cat is purring",
                "The bear is growling");


        //more info: https://ollama4j.github.io/ollama4j/apis-generate/generate-embeddings/
        final OllamaEmbedResponseModel embeddings = ollamaAPI.embed(new OllamaEmbedRequestModel("albertogg/multi-qa-minilm-l6-cos-v1", input));

        for (int i = 0; i < input.size(); i++) {
            PreparedStatement insertStmt = conn.prepareStatement("INSERT INTO documents (content, embedding) VALUES (?, ?)");
            insertStmt.setString(1, input.get(i));
            insertStmt.setObject(2, new PGvector(embeddings.getEmbeddings().get(i)));
            insertStmt.executeUpdate();
        }

        final String query = "growling bear";
        final List<Double> queryEmbedding = ollamaAPI.embed(new OllamaEmbedRequestModel("albertogg/multi-qa-minilm-l6-cos-v1", List.of(query))).getEmbeddings().getFirst();
        final double k = 60;

        PreparedStatement queryStmt = conn.prepareStatement(HYBRID_SQL);
        queryStmt.setObject(1, new PGvector(queryEmbedding));
        queryStmt.setObject(2, new PGvector(queryEmbedding));
        queryStmt.setString(3, query);
        queryStmt.setDouble(4, k);
        queryStmt.setDouble(5, k);
        ResultSet rs = queryStmt.executeQuery();
        while (rs.next()) {
            System.out.printf("document: %d, RRF score: %f%n", rs.getLong("id"), rs.getDouble("score"));
        }

        conn.close();
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
