package com.example;

import java.io.IOException;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpRequest.BodyPublishers;
import java.net.http.HttpResponse;
import java.net.http.HttpResponse.BodyHandlers;
import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.sql.Statement;
import java.util.ArrayList;
import java.util.List;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import com.pgvector.PGvector;

public class Example {
    public static void main(String[] args) throws IOException, InterruptedException, SQLException {
        String apiKey = System.getenv("OPENAI_API_KEY");
        if (apiKey == null) {
            System.out.println("Set OPENAI_API_KEY");
            System.exit(1);
        }

        Connection conn = DriverManager.getConnection("jdbc:postgresql://localhost:5432/pgvector_example");

        Statement setupStmt = conn.createStatement();
        setupStmt.executeUpdate("CREATE EXTENSION IF NOT EXISTS vector");
        setupStmt.executeUpdate("DROP TABLE IF EXISTS documents");

        PGvector.addVectorType(conn);

        Statement createStmt = conn.createStatement();
        createStmt.executeUpdate("CREATE TABLE documents (id bigserial PRIMARY KEY, content text, embedding vector(1536))");

        String[] input = {
            "The dog is barking",
            "The cat is purring",
            "The bear is growling"
        };
        List<float[]> embeddings = embed(input, apiKey);

        for (int i = 0; i < input.length; i++) {
            PreparedStatement insertStmt = conn.prepareStatement("INSERT INTO documents (content, embedding) VALUES (?, ?)");
            insertStmt.setString(1, input[i]);
            insertStmt.setObject(2, new PGvector(embeddings.get(i)));
            insertStmt.executeUpdate();
        }

        String query = "forest";
        float[] queryEmbedding = embed(new String[] {query}, apiKey).get(0);
        PreparedStatement neighborStmt = conn.prepareStatement("SELECT content FROM documents ORDER BY embedding <=> ? LIMIT 5");
        neighborStmt.setObject(1, new PGvector(queryEmbedding));
        ResultSet rs = neighborStmt.executeQuery();
        while (rs.next()) {
            System.out.println(rs.getString("content"));
        }

        conn.close();
    }

    private static List<float[]> embed(String[] input, String apiKey) throws IOException, InterruptedException {
        ObjectMapper mapper = new ObjectMapper();
        ObjectNode root = mapper.createObjectNode();
        for (String v : input) {
            root.withArray("input").add(v);
        }
        root.put("model", "text-embedding-3-small");
        String json = mapper.writeValueAsString(root);

        HttpClient client = HttpClient.newHttpClient();
        HttpRequest request = HttpRequest.newBuilder()
            .uri(URI.create("https://api.openai.com/v1/embeddings"))
            .header("Authorization", "Bearer " + apiKey)
            .header("Content-Type", "application/json")
            .POST(BodyPublishers.ofString(json))
            .build();
        HttpResponse<String> response = client.send(request, BodyHandlers.ofString());

        List<float[]> embeddings = new ArrayList<>();
        for (JsonNode n : mapper.readTree(response.body()).get("data")) {
            float[] embedding = new float[n.get("embedding").size()];
            int i = 0;
            for (JsonNode v : n.get("embedding")) {
                embedding[i++] = (float) v.asDouble();
            }
            embeddings.add(embedding);
        }
        return embeddings;
    }
}
