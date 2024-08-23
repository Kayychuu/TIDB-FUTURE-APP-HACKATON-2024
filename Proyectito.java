

package poo.proyectito;

import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.SQLException;

public class Proyectito {
    // Configuración de la conexión
    private static final String URL = "jdbc:mysql://gateway01.us-east-1.prod.aws.tidbcloud.com:3306/test";
    private static final String USER = "4AfmDMppLajE5rg.root";
    private static final String PASSWORD = "NAVUajR9PvVrkdqN";

    public static void main(String[] args) {
        Connection connection = null;

        try {
            // Establecer la conexión
            connection = DriverManager.getConnection(URL, USER, PASSWORD);
            if (connection != null) {
                System.out.println("Conexión exitosa a la base de datos!");
            }
        } catch (SQLException e) {
            e.printStackTrace();
            System.out.println("Error al conectar a la base de datos: " + e.getMessage());
        } finally {
            try {
                if (connection != null && !connection.isClosed()) {
                    connection.close();
                    System.out.println("Conexión cerrada.");
                }
            } catch (SQLException e) {
                e.printStackTrace();
            }
        }
    }
}