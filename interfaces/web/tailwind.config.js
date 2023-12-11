/** @type {import('tailwindcss').Config} */
import colors from 'tailwindcss/colors'


module.exports = {
    content: ["./templates/**/*.{html,js}"],
    theme: {
    extend: {
        ...colors,
    }
    },
    plugins: [],
}