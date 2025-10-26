const path = require("path");

module.exports = {
    devtool: "source-map",
    entry: ["./scaleforce/main.ts"],
    output: {
        path: path.join(__dirname, "/js"),
        filename: "[name].js",
    },
    resolve: {
        extensions: [".ts", ".js", ".tsx", ".json"]
    },
    module: {
        rules: [
            {
                test: /\.tsx?$/,
                loader: "ts-loader",
            },
        ],
    },
    plugins: []
}