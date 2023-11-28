/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Class.java to edit this template
 */
package ActividadesParejas;

import java.util.Scanner;

/**
 *
 * @author batoi
 */
public class Actividad292 {

    public static Scanner scanner;

    public static void main(String[] args) {
        scanner = new Scanner(System.in);
        System.out.print("Introduzca una palabra o frase: ");
        String frase = scanner.nextLine();
        int vocales = contadorVocales(frase);
        System.out.printf("La palabra o frase contiene %d vocales\n", vocales);
    }

    public static int contadorVocales(String frase) {
        String vocal="aeiou";
        int vocales = 0;
        for (int i = 0; i < frase.length(); i++) {
            for (int j = 0; j < vocal.length(); j++) {
                if (frase.charAt(i) == vocal.charAt(j)||frase.charAt(i) == vocal.toUpperCase().charAt(j)) {
                vocales++;
            }
            }
            

        }
        return vocales;
    }
}
